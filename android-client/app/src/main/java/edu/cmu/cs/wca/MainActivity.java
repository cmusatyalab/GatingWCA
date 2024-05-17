package edu.cmu.cs.wca;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.Drawable;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.VideoView;

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

import edu.cmu.cs.gabriel.camera.CameraCapture;
import edu.cmu.cs.gabriel.camera.ImageViewUpdater;
import edu.cmu.cs.gabriel.camera.YuvToJPEGConverter;
import edu.cmu.cs.gabriel.client.comm.ServerComm;
import edu.cmu.cs.gabriel.client.results.ErrorType;
import edu.cmu.cs.gabriel.protocol.Protos.InputFrame;
import edu.cmu.cs.gabriel.protocol.Protos.ResultWrapper;
import edu.cmu.cs.gabriel.protocol.Protos.PayloadType;
import edu.cmu.cs.wca.Protos.ToClientExtras;
import edu.cmu.cs.wca.Protos.ToServerExtras;
import edu.cmu.cs.wca.Protos.ZoomInfo;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final String VIDEO_NAME = "video";
    private static final String SOURCE = "wca_client";
    private static final int PORT = 9099;

    // Attempt to get the largest images possible. ImageAnalysis is limited to something below 1080p
    // according to this:
    // https://developer.android.com/reference/androidx/camera/core/ImageAnalysis.Builder#setTargetResolution(android.util.Size)
    private static final int WIDTH = 1920;
    private static final int HEIGHT = 1080;

    public static final String EXTRA_JWT_TOKEN = "edu.cmu.cs.wca.JWT_TOKEN";
    public static final String EXTRA_MEETING_NUMBER = "edu.cmu.cs.wca.MEETING_NUMBER";
    public static final String EXTRA_MEETING_PASSWORD = "edu.cmu.cs.wca.MEETING_PASSWORD";

    private static final String WCA_FSM_START = "WCA_FSM_START";
    private static final String WCA_FSM_END = "WCA_FSM_END";
    private ToServerExtras.ClientCmd reqCommand = ToServerExtras.ClientCmd.NO_CMD;
    private boolean logCompleted = false;
    private String step = WCA_FSM_START;
    private ServerComm serverComm;
    private YuvToJPEGConverter yuvToJPEGConverter;
    private CameraCapture cameraCapture;
    private TextToSpeech textToSpeech;
    private ImageViewUpdater instructionViewUpdater;
    private ImageView instructionImage;
    private ImageView readyView;
    private TextView readyTextView;
    private VideoView instructionVideo;
    private File videoFile;
    private final SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss-z", Locale.US);
    private final String LOGFILE = "THUMBSUP-" + sdf.format(new Date()) + ".txt";
    final ConcurrentLinkedDeque<String> logList = new ConcurrentLinkedDeque<>();
    private AtomicBoolean logProfiling = new AtomicBoolean(true);
    private FileWriter logFileWriter;
    private int inputFrameCount = 0;
    private ThumbsUpDetection thumbsUpDetector;
    private boolean readyForServer = false;
    private long currentStepStartTime = 0;

    private final ActivityResultLauncher<Intent> activityResultLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            new ActivityResultCallback<ActivityResult>() {
                @Override
                public void onActivityResult(ActivityResult result) {
                    ToServerExtras toServerExtras = ToServerExtras.newBuilder()
                            .setClientCmd(ToServerExtras.ClientCmd.ZOOM_STOP)
                            .build();

                    serverComm.send(
                            InputFrame.newBuilder().setExtras(pack(toServerExtras)).build(),
                            SOURCE,
                            /* wait */ true);
                }
            });

    private final Consumer<ResultWrapper> consumer = resultWrapper -> {
        try {
            ToClientExtras toClientExtras = ToClientExtras.parseFrom(
                    resultWrapper.getExtras().getValue());
            if (toClientExtras.getZoomResult() == ToClientExtras.ZoomResult.CALL_START) {
                ZoomInfo zoomInfo = toClientExtras.getZoomInfo();

                Intent intent = new Intent(this, ZoomActivity.class);
                intent.putExtra(EXTRA_JWT_TOKEN, zoomInfo.getJwtToken());
                intent.putExtra(EXTRA_MEETING_NUMBER, zoomInfo.getMeetingNumber());
                intent.putExtra(EXTRA_MEETING_PASSWORD, zoomInfo.getMeetingPassword());

                activityResultLauncher.launch(intent);
                return;
            } else if (toClientExtras.getZoomResult() == ToClientExtras.ZoomResult.EXPERT_BUSY) {
                runOnUiThread(() -> {
                    AlertDialog alertDialog = new AlertDialog.Builder(this)
                            .setTitle("Expert Busy")
                            .setMessage("The expert is currently helping someone else.")
                            .create();
                    alertDialog.show();
                });
            }

            if (step.equals(WCA_FSM_START)) {
                logList.add(TAG + ": Start: " + SystemClock.uptimeMillis() + "\n");
                inputFrameCount = 1;
                Log.i(TAG, "Profiling started.");
            }
            step = toClientExtras.getStep();
            if (step.equals(WCA_FSM_END) && !logCompleted) {
                logCompleted = true;
                logList.add(TAG + ": Total Input Frames: " + inputFrameCount + "\n");
                logList.add(TAG + ": Stop: " + SystemClock.uptimeMillis() + "\n");
                writeLog();
                readyForServer = false;
                runOnUiThread(() -> {
                    readyView.setVisibility(View.INVISIBLE);
                    readyTextView.setVisibility(View.INVISIBLE);
                });
                Log.i(TAG, "Profiling completed.");
            }

        } catch (InvalidProtocolBufferException e) {
            Log.e(TAG, "Protobuf parse error", e);
        }

        if (resultWrapper.getResultsCount() == 0) {
            return;
        }

        boolean hasVideo = false;
        for (ResultWrapper.Result result : resultWrapper.getResultsList()) {
            if (result.getPayloadType() == PayloadType.VIDEO) {
                hasVideo = true;
                break;
            }
        }

        // Load the user guidance (audio, image/video) from the result wrapper
        for (ResultWrapper.Result result : resultWrapper.getResultsList()) {
            if (result.getPayloadType() == PayloadType.TEXT) {
                if (!step.equals(WCA_FSM_END)) {
                    readyForServer = false;
                    currentStepStartTime = SystemClock.uptimeMillis();
                    runOnUiThread(() -> {
                        readyView.setVisibility(View.INVISIBLE);
                        readyTextView.setVisibility(View.VISIBLE);
                    });
                }

                ByteString dataString = result.getPayload();
                String speech = dataString.toStringUtf8();
                this.textToSpeech.speak(speech, TextToSpeech.QUEUE_FLUSH, null, null);
                Log.i(TAG, "Saying: " + speech);
            } else if ((result.getPayloadType() == PayloadType.IMAGE) && !hasVideo) {
                ByteString image = result.getPayload();
                instructionViewUpdater.accept(image);

                runOnUiThread(() -> {
                    instructionImage.setVisibility(View.VISIBLE);
                    instructionVideo.setVisibility(View.INVISIBLE);
                    instructionVideo.stopPlayback();
                });
            } else if (result.getPayloadType() == PayloadType.VIDEO) {
                try {
                    videoFile.delete();
                    videoFile.createNewFile();
                    FileOutputStream fos = new FileOutputStream(videoFile);
                    result.getPayload().writeTo(fos);
                    fos.close();

                    runOnUiThread(() -> {
                        instructionVideo.setVideoPath(videoFile.getPath());
                        instructionVideo.start();

                        instructionImage.setVisibility(View.INVISIBLE);
                        instructionVideo.setVisibility(View.VISIBLE);
                    });
                } catch (IOException e) {
                    Log.e(TAG, "video file failed", e);
                }
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        videoFile = new File(this.getCacheDir(), VIDEO_NAME);
        PreviewView viewFinder = findViewById(R.id.viewFinder);

        readyView = findViewById(R.id.readyView);
        readyTextView = findViewById(R.id.readyTextView);
        AssetManager assetManager = getAssets();
        try
        {
            InputStream ins = assetManager.open("thumbs_up.png");
            Drawable drawable = Drawable.createFromStream(ins, null);
            readyView.setImageDrawable(drawable);
            ins.close();
        }
        catch(IOException e) {
            e.printStackTrace();
        }

        instructionImage = findViewById(R.id.instructionImage);
        instructionViewUpdater = new ImageViewUpdater(instructionImage);

        instructionVideo = findViewById(R.id.instructionVideo);

        // from https://stackoverflow.com/a/8431374/859277
        instructionVideo.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
            @Override
            public void onPrepared(MediaPlayer mp) {
                mp.setLooping(true);
            }
        });

        File logFile = new File(getExternalFilesDir(null), LOGFILE);
        logFile.delete();
        logFile = new File(getExternalFilesDir(null), LOGFILE);
        try {
            logFileWriter = new FileWriter(logFile, true);
        } catch (IOException e) {
            e.printStackTrace();
        }

        Consumer<ErrorType> onDisconnect = errorType -> {
            Log.e(TAG, "Disconnect Error: " + errorType.name());
            finish();
        };
        serverComm = ServerComm.createServerComm(
                consumer, BuildConfig.GABRIEL_HOST, PORT, getApplication(), onDisconnect);

        TextToSpeech.OnInitListener onInitListener = status -> {
            if (status == TextToSpeech.ERROR) {
                Log.e(TAG, "TextToSpeech initialization failed with status " + status);
            }
            ToServerExtras toServerExtras = ToServerExtras.newBuilder().setStep(step).build();
            InputFrame inputFrame = InputFrame.newBuilder()
                    .setExtras(pack(toServerExtras))
                    .build();

            // We need to wait for textToSpeech to be initialized before asking for the first
            // instruction.
            serverComm.send(inputFrame, SOURCE, /* wait */ true);
        };
        this.textToSpeech = new TextToSpeech(getApplicationContext(), onInitListener);
        yuvToJPEGConverter = new YuvToJPEGConverter(this, 100);

        // Ask for runtime permission to record audio
        String[] extraPermissions = new String[]{
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO,
        };
        int recordAudioPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO);
        int cameraPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA);
        if (recordAudioPermission != PackageManager.PERMISSION_GRANTED || cameraPermission != PackageManager.PERMISSION_GRANTED) {
            ActivityResultCallback<Map<String, Boolean>> activityResultCallback = resultMap -> {
                if (resultMap.values().stream().allMatch(Boolean.TRUE::equals)) {
                    cameraCapture = new CameraCapture(this, analyzer, WIDTH, HEIGHT, viewFinder, CameraSelector.DEFAULT_BACK_CAMERA, false);
                } else {
                    Toast.makeText(this, "The user denied the required permissions.", Toast.LENGTH_LONG).show();
                }
            };

            ActivityResultLauncher<String[]> requestPermissionLauncher =
                    this.registerForActivityResult(
                            new ActivityResultContracts.RequestMultiplePermissions(), activityResultCallback);
            requestPermissionLauncher.launch(extraPermissions);
        } else {
            cameraCapture = new CameraCapture(this, analyzer, WIDTH, HEIGHT, viewFinder, CameraSelector.DEFAULT_BACK_CAMERA, false);
        }

        thumbsUpDetector = new ThumbsUpDetection(this);
        thumbsUpDetector.hands.setResultListener(
                handsResult -> {
                    if (!handsResult.multiHandLandmarks().isEmpty()) {
                        if (handsResult.timestamp() > currentStepStartTime &&
                                ThumbsUpDetection.detectThumbsUp(handsResult)) {
                            readyForServer = true;
                            runOnUiThread(() -> {
                                readyView.setVisibility(View.VISIBLE);
                                readyTextView.setVisibility(View.VISIBLE);
                            });
                        }
                    }
                });
        thumbsUpDetector.hands.setErrorListener((message, e) -> Log.e(TAG, "MediaPipe Hands error:" + message));
    }

    private void writeLog() {
        logProfiling.set(false);
        try {
            for (String logString: logList) {
                logFileWriter.write(logString);
            }
            logFileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Based on
    // https://github.com/protocolbuffers/protobuf/blob/master/src/google/protobuf/compiler/java/java_message.cc#L1387
    public static Any pack(ToServerExtras toServerExtras) {
        return Any.newBuilder()
                .setTypeUrl("type.googleapis.com/wca.ToServerExtras")
                .setValue(toServerExtras.toByteString())
                .build();
    }

    final private ImageAnalysis.Analyzer analyzer = new ImageAnalysis.Analyzer() {
        @Override
        public void analyze(@NonNull ImageProxy image) {
            // Ensure the packet is sent to the server if there is a client command
            boolean toWait = (reqCommand != ToServerExtras.ClientCmd.NO_CMD);
            if (step.equals(WCA_FSM_END) && !toWait) {
                image.close();
                return;
            }
            inputFrameCount++;

            if (readyForServer || toWait) {
                ToServerExtras.ClientCmd clientCmd = reqCommand;
                reqCommand = ToServerExtras.ClientCmd.NO_CMD;
                serverComm.sendSupplier(() -> {
                    ByteString jpegByteString = yuvToJPEGConverter.convert(image);

                    ToServerExtras toServerExtras = ToServerExtras.newBuilder()
                            .setStep(MainActivity.this.step)
                            .setClientCmd(clientCmd)
                            .build();

                    return InputFrame.newBuilder()
                            .setPayloadType(PayloadType.IMAGE)
                            .addPayloads(jpegByteString)
                            .setExtras(pack(toServerExtras))
                            .build();
                }, SOURCE, /* wait */ toWait);
            } else {
                Bitmap bitmapImage = BitmapFactory.decodeStream(
                        new ByteArrayInputStream(yuvToJPEGConverter.convert(image).toByteArray()));
                thumbsUpDetector.hands.send(bitmapImage, SystemClock.uptimeMillis());
            }

            // The image has either been sent or skipped. It is therefore safe to close the image.
            image.close();
        }
    };

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraCapture.shutdown();
    }

    public void startZoom(View view) {
        this.textToSpeech.speak("Calling expert now.",
                TextToSpeech.QUEUE_FLUSH, null, null);
        // Delay sending the client command until the next frame
        this.reqCommand = ToServerExtras.ClientCmd.ZOOM_START;
    }
}