using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using Unity.Sentis;
using Unity.Sentis.Layers;
using System.IO;
using Newtonsoft.Json;
using System.Text;

public class RunWhisper : MonoBehaviour
{
    Worker decoderEngine, encoderEngine, spectroEngine;
    const BackendType backend = BackendType.GPUCompute;

    public AudioClip audioClip;
    public SpeechRecognitionController speechRecognitionController;

    const int maxTokens = 100;
    const int END_OF_TEXT = 50257;
    const int START_OF_TRANSCRIPT = 50258;
    const int ENGLISH = 50259;
    const int TRANSCRIBE = 50359;
    const int START_TIME = 50364;

    int numSamples;
    float[] data;
    string[] tokens;
    int currentToken = 0;
    int[] outputTokens = new int[maxTokens];

    int[] whiteSpaceCharacters = new int[256];
    Tensor encodedAudio;
    bool transcribe = false;
    string outputString = "";
    const int maxSamples = 30 * 16000;

    void Start()
    {
        SetupWhiteSpaceShifts();
        GetTokens();

        Model decoder = ModelLoader.Load(Application.streamingAssetsPath + "/AudioDecoder_Tiny.sentis");
        Model encoder = ModelLoader.Load(Application.streamingAssetsPath + "/AudioEncoder_Tiny.sentis");
        Model spectro = ModelLoader.Load(Application.streamingAssetsPath + "/LogMelSepctro.sentis");

        decoderEngine = WorkerFactory.CreateWorker(backend, decoder);
        encoderEngine = WorkerFactory.CreateWorker(backend, encoder);
        spectroEngine = WorkerFactory.CreateWorker(backend, spectro);
    }

    public void Transcribe()
    {
        outputTokens[0] = START_OF_TRANSCRIPT;
        outputTokens[1] = ENGLISH;
        outputTokens[2] = TRANSCRIBE;
        outputTokens[3] = START_TIME;
        currentToken = 3;
        outputString = "";

        LoadAudio();
        EncodeAudio();
        transcribe = true;
    }

    void LoadAudio()
    {
        if (audioClip.frequency != 16000)
        {
            Debug.LogWarning($"AudioClip should be 16kHz. Current: {audioClip.frequency}");
            return;
        }

        numSamples = audioClip.samples;
        if (numSamples > maxSamples)
        {
            Debug.LogWarning($"AudioClip too long: {numSamples / audioClip.frequency} seconds.");
            return;
        }

        data = new float[numSamples];
        audioClip.GetData(data, 0);

        if (numSamples < maxSamples)
        {
            float[] padded = new float[maxSamples];
            data.CopyTo(padded, 0);
            data = padded;
            numSamples = maxSamples;
        }
    }

    void GetTokens()
    {
        var jsonText = File.ReadAllText(Application.streamingAssetsPath + "/vocab.json");
        var vocab = JsonConvert.DeserializeObject<Dictionary<string, int>>(jsonText);
        tokens = new string[vocab.Count];
        foreach (var item in vocab)
            tokens[item.Value] = item.Key;
    }

    void EncodeAudio()
    {
        using var input = TensorFloat.Create(new TensorShape(1, numSamples), data);
        spectroEngine.Execute(input);
        var spectroOutput = spectroEngine.PeekOutput() as TensorFloat;
        encoderEngine.Execute(spectroOutput);
        encodedAudio = encoderEngine.PeekOutput();

        input.Dispose();
        spectroOutput.Dispose();
    }

    void Update()
    {
        if (!transcribe) return;
        if (currentToken >= outputTokens.Length - 1) return;

        using var tokensSoFar = new Tensor(new TensorShape(1, outputTokens.Length), outputTokens);
        var inputs = new Dictionary<string, Tensor>
        {
            { "encoded_audio", encodedAudio },
            { "tokens", tokensSoFar }
        };

        decoderEngine.Execute(inputs);

        var tokensOutTensor = decoderEngine.PeekOutput();
        float[] tokensOut = tokensOutTensor.AsFloats(); // convert to float array

        int ID = ArgMax(tokensOut);
        outputTokens[++currentToken] = ID;

        if (ID == END_OF_TEXT)
            transcribe = false;
        else if (ID >= tokens.Length)
            outputString += $"(time={(ID - START_TIME) * 0.02f})";
        else
            outputString += GetUnicodeText(tokens[ID]);

        speechRecognitionController.onResponse.Invoke(outputString);
        Debug.Log(outputString);
    }

    int ArgMax(float[] arr)
    {
        float max = float.MinValue;
        int maxIdx = 0;
        for (int i = 0; i < arr.Length; i++)
            if (arr[i] > max) { max = arr[i]; maxIdx = i; }
        return maxIdx;
    }

    string GetUnicodeText(string text)
    {
        var bytes = Encoding.GetEncoding("ISO-8859-1").GetBytes(ShiftCharacterDown(text));
        return Encoding.UTF8.GetString(bytes);
    }

    string ShiftCharacterDown(string text)
    {
        string outText = "";
        foreach (char letter in text)
            outText += ((int)letter <= 256) ? letter : (char)whiteSpaceCharacters[(int)(letter - 256)];
        return outText;
    }

    void SetupWhiteSpaceShifts()
    {
        for (int i = 0, n = 0; i < 256; i++)
            if (IsWhiteSpace((char)i)) whiteSpaceCharacters[n++] = i;
    }

    bool IsWhiteSpace(char c) => !(('!' <= c && c <= '~') || ('¡' <= c && c <= '¬') || ('®' <= c && c <= 'ÿ'));

    private void OnDestroy()
    {
        decoderEngine?.Dispose();
        encoderEngine?.Dispose();
        spectroEngine?.Dispose();
    }
}
