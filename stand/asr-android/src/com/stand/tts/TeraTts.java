/*
 * Russian Voice Assistant modification for Changan A06 (C390).
 * Copyright (c) 2026 Tecrow.
 * Licensed under the PolyForm Noncommercial License 1.0.0 — noncommercial use only. See LICENSE.
 * Independent modification — not affiliated with or endorsed by Changan Automobile.
 */
package com.stand.tts;

import android.content.Context;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.util.Log;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import com.stand.tts.tera.TeraTTS;

/**
 * TeraTTS (ru_f2) backend for the native TTS engine (PiperCaTts). Pure-Java inference on the STOCK
 * onnxruntime 1.17.1 (we bundle only libonnxruntime4j_jni.so; libonnxruntime.so is the stock one).
 * Output is mono float @44100 → resampled to the player rate (24000) and packed to 16-bit PCM.
 * synthesize() is NOT thread-safe → guarded by LOCK. Model assets (~370 MB) unpack to filesDir once.
 */
public final class TeraTts {
    private static final String TAG = "TeraTts";
    private static final String ASSET_DIR = "tera";     // apk assets/tera: models/, unicode_indexer.json, styles/
    private static final String VOICE = "ru_f2";
    private static final int NATIVE_RATE = 44100;       // TeraTTS output rate
    private static final float SPEED = 0.9f;            // durationScale <1 = faster speech (user wanted quicker)
    private static volatile TeraTTS engine;
    private static Context appCtx;
    private static final Object LOCK = new Object();

    // Rendered speech is cached PER SENTENCE: in memory, and on disk so it survives a restart.
    // Fixed greetings are still pre-warmed at init, but only until the disk cache has them.
    // The memory tier is bounded to keep :tts RSS in check.
    private static final int CACHE_MAX = 48;
    // Disk budget for rendered speech, about four minutes of it.
    private static final long DISK_CACHE_BYTES = 32L << 20;
    private static final java.util.Map<String, byte[]> CACHE =
            java.util.Collections.synchronizedMap(new java.util.LinkedHashMap<String, byte[]>(64, 0.75f, true) {
                protected boolean removeEldestEntry(java.util.Map.Entry<String, byte[]> e) { return size() > CACHE_MAX; }
            });
    // Fixed wake/idle prompts spoken by the assistant (see RuBridge.zh2ru: wake prompt → "Чем могу помочь").
    private static final String[] PREWARM = { "Чем могу помочь", "Пассажир, чем могу помочь",
                                              "Да", "До свидания" };

    private static String procName() {
        try {
            byte[] b = new byte[128];
            java.io.FileInputStream in = new java.io.FileInputStream("/proc/self/cmdline");
            int n = in.read(b); in.close();
            if (n <= 0) return "";
            int end = 0; while (end < n && b[end] != 0) end++;
            return new String(b, 0, end).trim();
        } catch (Throwable t) { return ""; }
    }
    private static boolean isTtsProcess() {
        return appCtx != null && procName().equals(appCtx.getPackageName() + ":tts");
    }
    /** Allowed in main (for --es say) and :tts (real engine). */
    public static boolean canSpeak() {
        if (ctx() == null) return false;
        String pn = procName(), pkg = appCtx.getPackageName();
        return pn.equals(pkg) || pn.equals(pkg + ":tts");
    }
    private static Context ctx() {
        if (appCtx != null) return appCtx;
        try {
            Class<?> at = Class.forName("android.app.ActivityThread");
            Object app = at.getMethod("currentApplication").invoke(null);
            if (app instanceof Context) appCtx = ((Context) app).getApplicationContext();
        } catch (Throwable ignored) {}
        return appCtx;
    }

    private static final java.util.concurrent.atomic.AtomicBoolean STARTED =
            new java.util.concurrent.atomic.AtomicBoolean(false);

    public static void init(final Context c) {
        appCtx = c.getApplicationContext();
        // init() is called once per PiperCaTts instance (config maps several engines to it); guard so
        // engine load + greeting prewarm run exactly ONCE, not 3× (which stormed the CPU at startup).
        if (isTtsProcess() && STARTED.compareAndSet(false, true)) new Thread(new Runnable() { public void run() {
            ensure();
            for (String g : PREWARM) {   // warm the engine + cache fixed greetings → instant on wake
                try { synthPcm16(g, RATE); } catch (Throwable ignored) {}
            }
            Log.i(TAG, "prewarmed " + CACHE.size() + " greeting phrase(s)");
        } }).start();
    }
    private static final int RATE = 24000;   // native TTS player rate (TtsAudioManager.SAMPLE_RATE_DEFAULT)

    private static synchronized boolean ensure() {
        if (engine != null) return true;
        try {
            File dir = new File(appCtx.getFilesDir(), ASSET_DIR);
            // pm install -r KEEPS filesDir, so a plain existence check would keep loading STALE models
            // after a model swap (fp16<->fp32, sampler step change) → broken/silent audio until pm clear.
            // Guard with a signature (name+size of every bundled model): on mismatch, wipe models/ and
            // re-unpack so a new APK's models always win without needing pm clear.
            if (needsUnpack(dir)) {
                Log.i(TAG, "unpacking Tera assets to " + dir + " (~370 MB; model set changed or missing)");
                deleteRec(new File(dir, "models"));          // drop stale models (incl. renamed sampler)
                unpackAssets(ASSET_DIR, dir);
                writeText(new File(dir, "models/.sig"), modelsSig());
            }
            // Runtime toggle for HW acceleration: `adb shell settings put global tera_nnapi 1` (then restart).
            try {
                com.stand.tts.tera.TeraTTS.USE_NNAPI = android.provider.Settings.Global.getInt(
                        appCtx.getContentResolver(), "tera_nnapi", 0) == 1;
            } catch (Throwable ignored) {}
            long t0 = System.currentTimeMillis();
            engine = new TeraTTS(dir.toPath(), VOICE);
            Log.i(TAG, "engine ready in " + (System.currentTimeMillis() - t0) + "ms, provider="
                    + com.stand.tts.tera.TeraTTS.PROVIDER_STATUS);
            return true;
        } catch (Throwable t) { Log.e(TAG, "ensure", t); return false; }
    }

    /** Synthesize Russian text to mono 16-bit LE PCM at targetRate (resampled from 44100). Empty on failure. */
    public static byte[] synthPcm16(String text, int targetRate) {
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        for (String sentence : sentences(ttsNormalize(text))) {
            byte[] pcm = synthSentence(sentence, targetRate);
            if (pcm.length == 0) return new byte[0];
            out.write(pcm, 0, pcm.length);
        }
        return out.toByteArray();
    }

    /** One normalized sentence: memory cache -> disk cache -> engine (then cached in both). */
    private static byte[] synthSentence(String t, int targetRate) {
        try {
            if (t.isEmpty() || ctx() == null) return new byte[0];
            String key = cacheKey(t, targetRate);
            byte[] hit = CACHE.get(key);
            if (hit == null) hit = readCached(key);
            if (hit != null) { remember(key, hit); return hit; }
            if (!ensure()) { Log.e(TAG, "synthSentence: engine not ready"); return new byte[0]; }
            long t0 = System.currentTimeMillis();
            float[] s;
            synchronized (LOCK) { s = engine.synthesize(t, SPEED); }
            byte[] pcm = pcm16(s, NATIVE_RATE, targetRate);
            Log.i(TAG, "synth " + pcm.length / 2 + " @" + targetRate + " (from " + NATIVE_RATE + ") in "
                    + (System.currentTimeMillis() - t0) + "ms: " + t);
            remember(key, pcm);
            writeCached(key, pcm);
            return pcm;
        } catch (Throwable e) { Log.e(TAG, "synthSentence", e); return new byte[0]; }
    }

    /** Sentence chunks of normalized text. Each is rendered and cached on its own, so a long reply
     *  starts playing after its first sentence and fixed sentences hit the cache inside variable replies. */
    private static String[] sentences(String normalized) {
        java.util.List<String> out = new java.util.ArrayList<String>();
        if (normalized != null) for (String part : normalized.split("(?<=[.!?…])\\s+")) {
            if (!part.trim().isEmpty()) out.add(part.trim());
        }
        return out.toArray(new String[0]);
    }

    /** Everything besides the text that changes how a sentence sounds: the lexicon and the dictionary. */
    private static volatile String accents;
    private static String accentVersion() {
        String a = accents;
        if (a == null) accents = a = com.stand.tts.tera.TeraTTS.ACCENT_VERSION + "." + assetLen(ASSET_DIR + "/ruaccent.bin");
        return a;
    }

    /** Stable id of one rendered sentence: text, rate, speed, voice and the stress data. */
    private static String cacheKey(String sentence, int rate) {
        try {
            byte[] d = java.security.MessageDigest.getInstance("SHA-1")
                    .digest((sentence + "@" + rate + "@" + SPEED + "@" + VOICE + "@" + accentVersion()).getBytes("UTF-8"));
            StringBuilder sb = new StringBuilder(40);
            for (byte b : d) sb.append(Character.forDigit((b >> 4) & 15, 16)).append(Character.forDigit(b & 15, 16));
            return sb.toString();
        } catch (Exception e) { throw new IllegalStateException(e); }
    }

    /** Float mono @from -> 16-bit LE PCM @to (linear resample, clipped). */
    private static byte[] pcm16(float[] s, int from, int to) {
        float[] rs = (from == to) ? s : resample(s, from, to);
        byte[] pcm = new byte[rs.length * 2];
        for (int i = 0; i < rs.length; i++) {
            int v = Math.round(rs[i] * 32767f);
            if (v > 32767) v = 32767; else if (v < -32768) v = -32768;
            pcm[i * 2] = (byte) (v & 0xff);
            pcm[i * 2 + 1] = (byte) ((v >> 8) & 0xff);
        }
        return pcm;
    }

    private static void remember(String key, byte[] pcm) {
        if (pcm.length <= 400000) CACHE.put(key, pcm);   // cache short phrases (~<8s) for instant repeat
    }
    /** Under models/ so a model swap wipes the rendered speech together with the stale models. */
    private static File cacheDir() {
        File dir = new File(appCtx.getFilesDir(), ASSET_DIR + "/models/cache");
        dir.mkdirs();
        return dir;
    }
    private static byte[] readCached(String key) {
        try {
            File f = new File(cacheDir(), key + ".pcm");
            if (f.isFile()) {
                f.setLastModified(System.currentTimeMillis());   // a used render survives the next trim
                return java.nio.file.Files.readAllBytes(f.toPath());
            }
        } catch (Throwable ignored) {}
        return null;
    }
    /** The stress data is part of the key, so a lexicon change simply stops matching: those renders
     *  become unreachable and age out of the budget instead of needing an explicit wipe. */
    private static void writeCached(String key, byte[] pcm) {
        try {
            File dir = cacheDir();
            java.nio.file.Files.write(new File(dir, key + ".pcm").toPath(), pcm);
            trimCache(dir, DISK_CACHE_BYTES);
        } catch (Throwable e) { Log.e(TAG, "cache write", e); }
    }
    /** Drop the least recently used renders until the directory fits the budget. */
    static int trimCache(File dir, long budget) {
        File[] files = dir == null ? null : dir.listFiles();
        if (files == null) return 0;
        long total = 0;
        for (File f : files) total += f.length();
        if (total <= budget) return 0;
        java.util.Arrays.sort(files, new java.util.Comparator<File>() {
            public int compare(File a, File b) { return Long.compare(a.lastModified(), b.lastModified()); }
        });
        int removed = 0;
        for (File f : files) {
            if (total <= budget) break;
            long size = f.length();
            if (f.delete()) { total -= size; removed++; }
        }
        return removed;
    }

    private static float[] resample(float[] in, int from, int to) {
        if (from == to || in.length == 0) return in;
        int outLen = (int) ((long) in.length * to / from);
        float[] out = new float[outLen];
        double step = (double) from / to;
        for (int i = 0; i < outLen; i++) {
            double pos = i * step; int j = (int) pos; double frac = pos - j;
            float a = in[j], b = (j + 1 < in.length) ? in[j + 1] : a;
            out[i] = (float) (a + (b - a) * frac);
        }
        return out;
    }

    /** Re-unpack if a key non-model file is missing OR the bundled model set (name+size of every file
     *  under assets/tera/models) differs from what was unpacked. Catches fp16<->fp32 and sampler-step
     *  swaps that reuse or change asset filenames — the plain existence check missed these. */
    private static boolean needsUnpack(File dir) {
        if (!new File(dir, "unicode_indexer.json").exists()
                || !new File(dir, "ruaccent.bin").exists()
                || !new File(dir, "styles/" + VOICE + "/style_ttl.bin").exists()) return true;
        String want = modelsSig();
        if (want.isEmpty()) return !new File(dir, "models/vocoder.onnx").exists(); // asset listing failed → fallback
        String have = readText(new File(dir, "models/.sig"));
        if (want.equals(have)) return false;
        Log.i(TAG, "models signature changed:\n  apk =" + want + "\n  disk=" + have);
        return true;
    }
    /** "name:size|..." (sorted) over assets/tera/models — cheap (openFd length, no full read; assets are zip-stored). */
    private static String modelsSig() {
        try {
            String base = ASSET_DIR + "/models";
            String[] list = appCtx.getAssets().list(base);
            if (list == null || list.length == 0) return "";
            java.util.Arrays.sort(list);
            StringBuilder sb = new StringBuilder();
            for (String n : list) { if (n.equals(".sig")) continue; sb.append(n).append(':').append(assetLen(base + "/" + n)).append('|'); }
            return sb.toString();
        } catch (Throwable t) { return ""; }
    }
    private static long assetLen(String path) {
        try { android.content.res.AssetFileDescriptor fd = appCtx.getAssets().openFd(path);
              long n = fd.getLength(); fd.close(); return n; }
        catch (Throwable t) { return -1; }
    }
    private static void deleteRec(File f) {
        try { if (f.isDirectory()) { File[] ch = f.listFiles(); if (ch != null) for (File c : ch) deleteRec(c); }
              f.delete(); } catch (Throwable ignored) {}
    }
    private static String readText(File f) {
        try { byte[] b = new byte[(int) f.length()]; java.io.FileInputStream in = new java.io.FileInputStream(f);
              int off = 0, n; while (off < b.length && (n = in.read(b, off, b.length - off)) > 0) off += n; in.close();
              return new String(b, 0, off, "UTF-8"); }
        catch (Throwable t) { return ""; }
    }
    private static void writeText(File f, String s) {
        try { f.getParentFile().mkdirs(); FileOutputStream o = new FileOutputStream(f);
              o.write(s.getBytes("UTF-8")); o.close(); }
        catch (Throwable t) { Log.e(TAG, "writeText " + f, t); }
    }

    private static void unpackAssets(String assetDir, File outDir) throws Exception {
        AssetManager am = appCtx.getAssets();
        String[] list = am.list(assetDir);
        if (list == null || list.length == 0) {   // file
            outDir.getParentFile().mkdirs();
            InputStream in = am.open(assetDir);
            OutputStream out = new FileOutputStream(outDir);
            byte[] b = new byte[1 << 16]; int n;
            while ((n = in.read(b)) > 0) out.write(b, 0, n);
            in.close(); out.close();
            return;
        }
        outDir.mkdirs();
        for (String name : list) unpackAssets(assetDir + "/" + name, new File(outDir, name));
    }

    // ---- Direct playback path (replaces the former PiperTts.speak): used to voice our own RU
    //      responses/tips (RuBridge.onTtsText / onTipText / StandNluReceiver). Synthesizes via
    //      TeraTTS (synthPcm16, cached) and streams the PCM through AudioTrack. -------------------
    private static volatile AudioTrack track;

    public static void speak(final String text) { speak(text, null); }

    /** Synthesize Russian text and play it on a background thread; run onDone after playback. */
    public static void speak(final String text, final Runnable onDone) {
        if (text == null || text.trim().isEmpty()) { if (onDone != null) runQuiet(onDone); return; }
        if (!canSpeak()) { if (onDone != null) runQuiet(onDone); return; }   // wrong process → skip
        new Thread(new Runnable() { public void run() {
            try {
                speakStreaming(text);
            } catch (Throwable t) { Log.e(TAG, "speak", t); }
            finally { if (onDone != null) runQuiet(onDone); }
        }}).start();
    }
    private static void runQuiet(Runnable r) { try { r.run(); } catch (Throwable ignored) {} }

    /** Play sentence by sentence: the first one starts the track, the rest are synthesized while it
     *  plays. Synthesis is faster than playback, so the queue stays ahead and the speech is continuous. */
    private static synchronized void speakStreaming(String text) throws InterruptedException {
        String[] parts = sentences(ttsNormalize(text));
        if (parts.length == 0) return;
        byte[] pcm = synthSentence(parts[0], RATE);
        if (pcm.length == 0) return;
        int min = AudioTrack.getMinBufferSize(RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
        AudioTrack at = new AudioTrack(AudioManager.STREAM_MUSIC, RATE,
                AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT,
                Math.max(min, RATE * 2 * 20), AudioTrack.MODE_STREAM);   // 20 s: a whole sentence is queued at once
        track = at;
        try {
            at.play();
            long frames = 0;
            for (int i = 0; i < parts.length; i++) {
                if (i > 0) pcm = synthSentence(parts[i], RATE);
                if (pcm.length == 0) break;
                frames += pcm.length / 2;
                at.write(pcm, 0, pcm.length);
            }
            long deadline = System.currentTimeMillis() + frames * 1000L / RATE + 2000;
            while ((at.getPlaybackHeadPosition() & 0xffffffffL) < frames
                    && System.currentTimeMillis() < deadline) Thread.sleep(10);
            Log.i(TAG, "playback done (" + frames + " frames @" + RATE + ", " + parts.length + " sentence(s))");
        } finally {
            try { at.stop(); } catch (Throwable ignored) {}
            at.release();
        }
    }

    // ---- Text normalization before neural synthesis (moved here from the former PiperTts):
    //   - drop the degree sign ("26 °C" -> "26"); expand units with RU number agreement
    //     ("30%" -> "30 процентов", "5 км" -> "5 километров"); rephrase masculine-past
    //     confirmations as first-person present ("Установил" -> "Ставлю"). ---------------------
    static String ttsNormalize(String t) {
        if (t == null) return "";
        String s = t.replaceAll("[°℃]\\s?[CcСс]?", "");          // degree unit
        s = expandUnit(s, "%",  "процент", "процента", "процентов");
        s = expandUnit(s, "км", "километр", "километра", "километров");
        s = femPresent(s);
        return s.replaceAll("\\s{2,}", " ").trim();
    }

    /** Replace "<number><unit>" with the number + the RU-agreeing word (1 процент / 2 процента / 5 процентов). */
    private static String expandUnit(String s, String unit, String one, String few, String many) {
        try {
            java.util.regex.Matcher m = java.util.regex.Pattern
                .compile("(\\d+)\\s*" + java.util.regex.Pattern.quote(unit) + "(?![А-Яа-яA-Za-z])").matcher(s);
            StringBuffer sb = new StringBuffer();
            while (m.find()) {
                String form;
                try { form = plural(Integer.parseInt(m.group(1)), one, few, many); }
                catch (Throwable e) { form = many; }
                m.appendReplacement(sb, java.util.regex.Matcher.quoteReplacement(m.group(1) + " " + form));
            }
            m.appendTail(sb);
            return sb.toString();
        } catch (Throwable e) { return s; }
    }

    /** Russian quantity agreement: 1 -> one, 2..4 -> few, else many (with the 11..14 exception). */
    private static String plural(int n, String one, String few, String many) {
        int m100 = ((n % 100) + 100) % 100, m10 = m100 % 10;
        if (m100 >= 11 && m100 <= 14) return many;
        if (m10 == 1) return one;
        if (m10 >= 2 && m10 <= 4) return few;
        return many;
    }

    /** Rephrase a first-person past confirmation as first-person present (gender-neutral):
     *  "Установил"/"Установила" -> "Ставлю". Whole words only, and only the masculine/feminine singular.
     *
     *  The previous version replaced substrings in map order, so the masculine entry matched the feminine
     *  form first and left its ending behind: "Установила" -> "Ставлюа", "Включила" -> "Включаюа". Matching
     *  substrings also rewrote unrelated words: "Спасибо, вы открыли мне новое" -> "вы открываюи мне новое".
     *  One entry per stem plus an optional "а" fixes both; the plural "открыли" is someone else's action
     *  and is deliberately left alone. */
    private static String femPresent(String s) {
        for (String[] p : PRESENT) {
            s = s.replaceAll("(?<![А-Яа-яЁё])" + p[0] + "а?(?![А-Яа-яЁё])", p[1]);
            s = s.replaceAll("(?<![А-Яа-яЁё])" + deCap(p[0]) + "а?(?![А-Яа-яЁё])", deCap(p[1]));
        }
        return s;
    }
    /** Masculine singular stem (the feminine adds "а") -> first-person present. */
    private static final String[][] PRESENT = {
        {"Установил","Ставлю"},{"Открыл","Открываю"},{"Закрыл","Закрываю"},
        {"Включил","Включаю"},{"Выключил","Выключаю"},{"Отключил","Отключаю"},
        {"Поднял","Поднимаю"},{"Опустил","Опускаю"},{"Увеличил","Увеличиваю"},
        {"Уменьшил","Уменьшаю"},{"Переключил","Переключаю"},{"Настроил","Настраиваю"},
        {"Запустил","Запускаю"},{"Сделал","Делаю"},{"Готов","Готова"}
    };
    private static String deCap(String w) {
        return w.isEmpty() ? w : Character.toLowerCase(w.charAt(0)) + w.substring(1);
    }

    private TeraTts() {}
}