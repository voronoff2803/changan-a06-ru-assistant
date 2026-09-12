/*
 * Russian Voice Assistant modification for Changan A06 (C390).
 * Copyright (c) 2026 Tecrow.
 * Licensed under the PolyForm Noncommercial License 1.0.0 — noncommercial use only. See LICENSE.
 * Independent modification — not affiliated with or endorsed by Changan Automobile.
 */
package com.stand.tts.tera;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.NNAPIFlags;
import java.util.EnumSet;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * TeraTTSv2 (TeraSpace/TeraTTSv2) — движок синтеза русской речи на чистом
 * onnxruntime Java API. Портировано с эталонного teratts.py (lite-вариант,
 * без ruaccent: ударения ставятся вручную знаком '+' перед ударной гласной,
 * либо модель угадывает их сама).
 *
 * Пайплайн: текст -> id символов (unicode_indexer.json) ->
 *   text_encoder -> duration_predictor -> sampler (8 шагов диффузии запечены
 *   в граф) -> vocoder -> float PCM 44100 Гц.
 *
 * Требует onnxruntime >= 1.16 (проверено на 1.17.1 — версия штатной
 * libonnxruntime на Changan A06).
 */
public class TeraTTS implements AutoCloseable {
    /** Hardware acceleration: try the NNAPI EP (NPU/GPU) at session build. Set by the wrapper before
     *  construction (from a runtime flag). Falls back to CPU per-op automatically; PROVIDER_STATUS
     *  records what actually engaged so the caller can log it. */
    public static volatile boolean USE_NNAPI = false;
    public static volatile String PROVIDER_STATUS = "cpu";
    /** Версия данных ударения: входит в ключ кеша готовой речи (см. TeraTts). */
    public static final String ACCENT_VERSION = TeraAccents.VERSION;
    public static final int SAMPLE_RATE = 44100;
    private static final int SAMPLES_PER_FRAME = 3072;
    private static final float BASE_SPEED = 1.05f;
    private static final long SEED = 1234L;
    private static final float GUIDANCE = 3.0f;

    private final OrtEnvironment env;
    private final OrtSession textEncoder;
    private final OrtSession durationPredictor;
    private final OrtSession sampler;
    private final OrtSession vocoder;
    private final int[] charTable; // 65536: unicode codepoint -> token id (-1 = нет)
    private final float[][][] styleTtl; // [1][50][256]
    private final float[][][] styleDp;  // [1][8][16]
    private final RuAccentDict accentDict; // частотный словарь ударений (assets/tera/ruaccent.bin; может быть null)

    /**
     * @param assetsDir каталог assets/ пакета (models/, styles/, unicode_indexer.json)
     * @param voice     имя голоса, например "ru_f2" (нужен styles/<voice>/)
     */
    public TeraTTS(Path assetsDir, String voice) throws OrtException, IOException {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        // big.LITTLE: half the cores = the big cluster; more threads thrash the LITTLE cores (slower).
        opts.setIntraOpNumThreads(Math.max(2, Runtime.getRuntime().availableProcessors() / 2));
        try { opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT); } catch (Throwable ignored) {}
        // RAM trim: drop the big pre-allocated CPU arena + memory-pattern planning. Output is identical;
        // only the allocation strategy changes (a touch more malloc/free per synth for less steady RAM).
        try { opts.setCPUArenaAllocator(false); } catch (Throwable ignored) {}
        try { opts.setMemoryPatternOptimization(false); } catch (Throwable ignored) {}
        if (USE_NNAPI) {   // hardware EP (NPU/GPU); unsupported ops fall back to CPU automatically
            try { opts.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16)); PROVIDER_STATUS = "nnapi"; }
            catch (Throwable t) { PROVIDER_STATUS = "cpu (nnapi unavailable: " + t.getMessage() + ")"; }
        } else { PROVIDER_STATUS = "cpu"; }
        Path models = assetsDir.resolve("models");
        textEncoder = env.createSession(models.resolve("text_encoder.onnx").toString(), opts);
        durationPredictor = env.createSession(models.resolve("duration_predictor.onnx").toString(), opts);
        sampler = env.createSession(models.resolve("sampler_distilled_cfg3_4step.onnx").toString(), opts);
        vocoder = env.createSession(models.resolve("vocoder.onnx").toString(), opts);
        charTable = loadIndexer(assetsDir.resolve("unicode_indexer.json"));
        Path style = assetsDir.resolve("styles").resolve(voice);
        styleTtl = reshape3(loadFloats(style.resolve("style_ttl.bin"), 50 * 256), 50, 256);
        styleDp = reshape3(loadFloats(style.resolve("style_dp.bin"), 8 * 16), 8, 16);
        Path dictFile = assetsDir.resolve("ruaccent.bin");   // optional frequency accent dict (~2 MB)
        RuAccentDict d = null;
        try { if (Files.exists(dictFile)) d = new RuAccentDict(dictFile); } catch (Throwable ignored) {}
        accentDict = d;
    }

    /**
     * Синтез одной реплики.
     *
     * @param text          русский текст; цифры разворачиваются в слова,
     *                      ударение можно задать '+': "зам+ок на двер+и"
     * @param durationScale 1.0 — обычный темп; больше — медленнее
     * @return mono float PCM [-1..1] @ 44100 Гц
     */
    public float[] synthesize(String text, float durationScale) throws OrtException {
        // Ударения проставляются ТОЛЬКО здесь, на входе синтеза (лексикон TeraAccents) —
        // UI-субтитр движка не касается и остаётся без '+'.
        String prepared = TeraAccents.accentize(prepareText(text), accentDict);
        String modelText = Normalizer.normalize(prepared, Normalizer.Form.NFKD);
        String durationText = modelText.replace("+", "");

        long[] textIds = encode(modelText);
        long[] durIds = encode(durationText);

        Map<String, OnnxTensor> encIn = new HashMap<String, OnnxTensor>();
        Map<String, OnnxTensor> durIn = new HashMap<String, OnnxTensor>();
        Map<String, OnnxTensor> samplerIn = new HashMap<String, OnnxTensor>();
        OrtSession.Result encOut = null, durOut = null, samplerOut = null, vocoderOut = null;
        try {
            encIn.put("text_ids", OnnxTensor.createTensor(env, new long[][]{textIds}));
            encIn.put("style_ttl", OnnxTensor.createTensor(env, styleTtl));
            encIn.put("text_mask", OnnxTensor.createTensor(env, onesMask(textIds.length)));
            encOut = textEncoder.run(encIn);

            durIn.put("text_ids", OnnxTensor.createTensor(env, new long[][]{durIds}));
            durIn.put("style_dp", OnnxTensor.createTensor(env, styleDp));
            durIn.put("text_mask", OnnxTensor.createTensor(env, onesMask(durIds.length)));
            durOut = durationPredictor.run(durIn);
            float rawDuration = firstFloat(durOut);

            float durationSeconds = rawDuration * durationScale / BASE_SPEED;
            if (!(durationSeconds > 0f) || Float.isInfinite(durationSeconds)) {
                throw new IllegalStateException("duration predictor returned non-positive duration");
            }
            int latentLength = Math.max(1,
                    (int) Math.ceil(durationSeconds * SAMPLE_RATE / (double) SAMPLES_PER_FRAME));

            samplerIn.put("initial_latent",
                    OnnxTensor.createTensor(env, gaussianLatent(latentLength)));
            samplerIn.put("text_emb", (OnnxTensor) encOut.get(0));
            samplerIn.put("style_ttl", OnnxTensor.createTensor(env, styleTtl));
            samplerIn.put("latent_mask", OnnxTensor.createTensor(env, onesMask(latentLength)));
            samplerIn.put("text_mask", OnnxTensor.createTensor(env, onesMask(textIds.length)));
            samplerIn.put("guidance", OnnxTensor.createTensor(env, new float[]{GUIDANCE}));
            samplerOut = sampler.run(samplerIn);

            Map<String, OnnxTensor> vocoderIn = new HashMap<String, OnnxTensor>();
            vocoderIn.put("latent", (OnnxTensor) samplerOut.get(0));
            vocoderOut = vocoder.run(vocoderIn);
            float[][] waveform = (float[][]) vocoderOut.get(0).getValue();

            int maxSamples = Math.min(waveform[0].length,
                    Math.round(durationSeconds * SAMPLE_RATE));
            float[] audio = new float[maxSamples];
            System.arraycopy(waveform[0], 0, audio, 0, maxSamples);
            return audio;
        } finally {
            // text_emb/latent берутся из Result-ов, поэтому Result закрываем последним
            for (OnnxTensor t : encIn.values()) t.close();
            for (OnnxTensor t : durIn.values()) t.close();
            for (Map.Entry<String, OnnxTensor> e : samplerIn.entrySet()) {
                if (!"text_emb".equals(e.getKey())) e.getValue().close();
            }
            if (vocoderOut != null) vocoderOut.close();
            if (samplerOut != null) samplerOut.close();
            if (durOut != null) durOut.close();
            if (encOut != null) encOut.close();
        }
    }

    // ---------- подготовка текста (порт normalize_text из teratts.py, без ruaccent) ----------

    /** NFC -> пробелы после пунктуации -> цифры в слова -> фильтр словаря -> теги <ru>. */
    String prepareText(String raw) {
        if (raw == null || raw.trim().isEmpty()) {
            throw new IllegalArgumentException("text must not be empty");
        }
        String text = Normalizer.normalize(raw, Normalizer.Form.NFC);
        text = text.replace("—", "-").replace("–", "-");
        text = addPunctuationSpaces(text);
        text = addDigitLetterSpaces(text);
        text = TeraNumbers.expandNumbers(text);
        text = skipUnsupported(text);
        if (!text.contains("<ru>") && !text.contains("<en>")) {
            text = "<ru>" + text + "</ru>";
        }
        return text;
    }

    private static String addPunctuationSpaces(String text) {
        StringBuilder sb = new StringBuilder(text.length() + 16);
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            sb.append(c);
            if (",.!?;:…".indexOf(c) >= 0 && i + 1 < text.length()) {
                char next = text.charAt(i + 1);
                boolean decimalInner = (c == '.' || c == ',')
                        && i > 0 && Character.isDigit(text.charAt(i - 1))
                        && Character.isDigit(next);
                if (next != ' ' && next != '<' && !decimalInner) {
                    sb.append(' ');
                }
            }
        }
        return sb.toString();
    }

    private static String addDigitLetterSpaces(String text) {
        StringBuilder sb = new StringBuilder(text.length() + 8);
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (i > 0 && Character.isDigit(text.charAt(i - 1)) && Character.isLetter(c)) {
                sb.append(' ');
            }
            sb.append(c);
        }
        return sb.toString();
    }

    /** Выкидывает символы, которых нет в словаре модели (сравнение по NFKD-разложению). */
    private String skipUnsupported(String text) {
        StringBuilder sb = new StringBuilder(text.length());
        for (int i = 0; i < text.length(); ) {
            int cp = text.codePointAt(i);
            String decomposed = Normalizer.normalize(
                    new String(Character.toChars(cp)), Normalizer.Form.NFKD);
            boolean ok = decomposed.length() > 0;
            for (int j = 0; ok && j < decomposed.length(); j++) {
                char d = decomposed.charAt(j);
                ok = d < 65536 && charTable[d] >= 0;
            }
            if (ok) sb.appendCodePoint(cp);
            i += Character.charCount(cp);
        }
        return sb.toString();
    }

    private long[] encode(String text) {
        long[] ids = new long[text.length()];
        int n = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            int token = c < 65536 ? charTable[c] : -1;
            if (token < 0) {
                throw new IllegalArgumentException(
                        "unsupported character U+" + Integer.toHexString(c));
            }
            ids[n++] = token;
        }
        if (n == 0) throw new IllegalArgumentException("text produced no tokens");
        return ids;
    }

    // ---------- тензорные утилиты ----------

    private static float[][][] onesMask(int length) {
        float[][][] mask = new float[1][1][length];
        java.util.Arrays.fill(mask[0][0], 1.0f);
        return mask;
    }

    private static float[][][] gaussianLatent(int frames) {
        Random rng = new Random(SEED);
        float[][][] latent = new float[1][144][frames];
        for (int c = 0; c < 144; c++) {
            for (int t = 0; t < frames; t++) {
                latent[0][c][t] = (float) rng.nextGaussian();
            }
        }
        return latent;
    }

    private static float firstFloat(OrtSession.Result result) throws OrtException {
        Object value = result.get(0).getValue();
        if (value instanceof float[]) return ((float[]) value)[0];
        if (value instanceof float[][]) return ((float[][]) value)[0][0];
        throw new IllegalStateException("unexpected duration output: " + value.getClass());
    }

    private static float[][][] reshape3(float[] flat, int d1, int d2) {
        float[][][] out = new float[1][d1][d2];
        for (int i = 0; i < d1; i++) {
            System.arraycopy(flat, i * d2, out[0][i], 0, d2);
        }
        return out;
    }

    // ---------- загрузка ассетов ----------

    private static int[] loadIndexer(Path path) throws IOException {
        String json = new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
        int[] table = new int[65536];
        int idx = 0, value = 0, sign = 1;
        boolean inNumber = false;
        for (int i = 0; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '-') {
                sign = -1;
                inNumber = true;
            } else if (c >= '0' && c <= '9') {
                value = value * 10 + (c - '0');
                inNumber = true;
            } else if (inNumber) {
                if (idx >= 65536) throw new IOException("indexer has too many entries");
                table[idx++] = sign * value;
                value = 0;
                sign = 1;
                inNumber = false;
            }
        }
        if (inNumber && idx < 65536) table[idx++] = sign * value;
        if (idx != 65536) throw new IOException("unicode_indexer.json must have 65536 entries, got " + idx);
        return table;
    }

    private static float[] loadFloats(Path path, int expected) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        if (bytes.length != expected * 4) {
            throw new IOException(path + ": expected " + expected * 4 + " bytes, got " + bytes.length);
        }
        float[] out = new float[expected];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out);
        return out;
    }

    @Override
    public void close() throws OrtException {
        textEncoder.close();
        durationPredictor.close();
        sampler.close();
        vocoder.close();
    }
}