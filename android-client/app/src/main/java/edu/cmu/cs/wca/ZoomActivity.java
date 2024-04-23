package edu.cmu.cs.wca;

import android.app.Service;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.Display;
import android.view.LayoutInflater;
import android.view.View;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import us.zoom.sdk.CameraDevice;
import us.zoom.sdk.InMeetingEventHandler;
import us.zoom.sdk.InMeetingService;
import us.zoom.sdk.InMeetingServiceListener;
import us.zoom.sdk.InMeetingUserInfo;
import us.zoom.sdk.InMeetingVideoController;
import us.zoom.sdk.JoinMeetingOptions;
import us.zoom.sdk.JoinMeetingParams;
import us.zoom.sdk.MeetingError;
import us.zoom.sdk.MeetingParameter;
import us.zoom.sdk.MeetingService;
import us.zoom.sdk.MeetingServiceListener;
import us.zoom.sdk.MeetingSettingsHelper;
import us.zoom.sdk.MeetingStatus;
import us.zoom.sdk.MobileRTCVideoUnitRenderInfo;
import us.zoom.sdk.MobileRTCVideoView;
import us.zoom.sdk.MobileRTCVideoViewManager;
import us.zoom.sdk.ZoomError;
import us.zoom.sdk.ZoomSDK;
import us.zoom.sdk.ZoomSDKInitParams;
import us.zoom.sdk.ZoomSDKInitializeListener;

public class ZoomActivity extends AppCompatActivity {
    private static final String TAG = "ZoomActivity";
    private static final String DISPLAY_NAME = "Gabriel User";
    private static final String ZOOM_DOMAIN = "zoom.us";

    private ZoomSDK zoomSDK;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        zoomSDK = ZoomSDK.getInstance();
        setContentView(R.layout.activity_zoom);

        if (!zoomSDK.isInitialized()) {
            Intent intent = getIntent();
            ZoomSDKInitParams initParams = new ZoomSDKInitParams();
            initParams.jwtToken = intent.getStringExtra(MainActivity.EXTRA_JWT_TOKEN);
            Log.w(TAG, "Zoom SDK Init: Got JWT token = " + initParams.jwtToken);
            initParams.domain = ZOOM_DOMAIN;
            zoomSDK.initialize(this, zoomSDKInitializeListener, initParams);
        }
        else if (zoomSDK.getMeetingService().getMeetingStatus() !=
                MeetingStatus.MEETING_STATUS_INMEETING) {
            startMeeting();
        } else {
            setupView();
        }
    }

    public void stopMeeting(View view) {
        zoomSDK.getInMeetingService().leaveCurrentMeeting(true);
    }

    private void startMeeting() {
        MeetingService meetingService = zoomSDK.getMeetingService();

        meetingService.addListener(meetingServiceListener);
        MeetingSettingsHelper meetingSettingsHelper = zoomSDK.getMeetingSettingsHelper();
        meetingSettingsHelper.setCustomizedMeetingUIEnabled(true);
        meetingSettingsHelper.setAutoConnectVoIPWhenJoinMeeting(false);

        zoomSDK.getInMeetingService().addListener(inMeetingServiceListener);

        Intent intent = getIntent();
        JoinMeetingParams params = new JoinMeetingParams();
        params.displayName = DISPLAY_NAME;
        params.meetingNo = intent.getStringExtra(MainActivity.EXTRA_MEETING_NUMBER);
        params.password = intent.getStringExtra(MainActivity.EXTRA_MEETING_PASSWORD);
        meetingService.joinMeetingWithParams(ZoomActivity.this, params, new JoinMeetingOptions());
    }

    private final ZoomSDKInitializeListener zoomSDKInitializeListener = new ZoomSDKInitializeListener() {
        @Override
        public void onZoomSDKInitializeResult(int errorCode, int internalErrorCode) {
            if (errorCode != ZoomError.ZOOM_ERROR_SUCCESS) {
                Toast.makeText(ZoomActivity.this, "Failed to initialize Zoom SDK. Error: " +
                                errorCode + ", internalErrorCode=" + internalErrorCode,
                        Toast.LENGTH_LONG).show();
            } else {
                startMeeting();
            }
        }

        @Override
        public void onZoomAuthIdentityExpired() {
            Toast.makeText(ZoomActivity.this, "Zoom SDK authentication identity expired.",
                    Toast.LENGTH_LONG).show();
        }
    };

    private final MeetingServiceListener meetingServiceListener = new MeetingServiceListener() {
        @Override
        public void onMeetingStatusChanged(MeetingStatus meetingStatus, int errorCode,
                                           int internalErrorCode) {
            Log.i(TAG, "onMeetingStatusChanged, meetingStatus=" + meetingStatus + ", errorCode=" +
                    errorCode + ", internalErrorCode=" + internalErrorCode);

            if(meetingStatus == MeetingStatus.MEETING_STATUS_FAILED &&
                    errorCode == MeetingError.MEETING_ERROR_CLIENT_INCOMPATIBLE) {
                Toast.makeText(ZoomActivity.this, "Version of ZoomSDK is too low!",
                        Toast.LENGTH_LONG).show();
            } else if (meetingStatus == MeetingStatus.MEETING_STATUS_INMEETING) {
                Log.i(TAG, "Meeting URL:" + zoomSDK.getMeetingService().getCurrentMeetingUrl());

                InMeetingService inMeetingService = zoomSDK.getInMeetingService();
                inMeetingService.setPlayChimeOnOff(false);
                inMeetingService.getInMeetingAudioController().connectAudioWithVoIP();
                inMeetingService.getInMeetingAudioController().muteMyAudio(false);
                InMeetingVideoController inMeetingVideoController = inMeetingService
                        .getInMeetingVideoController();
                for (CameraDevice cameraDevice : inMeetingVideoController.getCameraDeviceList()) {
                    if (cameraDevice.getCameraType() == CameraDevice.CAMERA_TYPE_BACK) {
                        inMeetingVideoController.switchCamera(cameraDevice.getDeviceId());
                    }
                }

                setupView();
            }
        }

        @Override
        public void onMeetingParameterNotification(MeetingParameter meetingParameter) {

        }
    };

    private void setupView() {
        LayoutInflater inflater = getLayoutInflater();
        View meetingVideo = inflater.inflate(R.layout.meeting_video, null);
        MobileRTCVideoView mobileRTCVideoView = meetingVideo.findViewById(R.id.videoView);

        FrameLayout meetingVideoView = findViewById(R.id.meetingVideoView);
        meetingVideoView.addView(meetingVideo);

        MobileRTCVideoUnitRenderInfo renderInfo = new MobileRTCVideoUnitRenderInfo(
                0, 0, 100, 100);

        InMeetingService inMeetingService = zoomSDK.getInMeetingService();
        InMeetingUserInfo myUserInfo = inMeetingService.getMyUserInfo();
        MobileRTCVideoViewManager mobileRTCVideoViewManager =
                mobileRTCVideoView.getVideoViewManager();
        mobileRTCVideoViewManager.addAttendeeVideoUnit(myUserInfo.getUserId(), renderInfo);

        renderInfo = new MobileRTCVideoUnitRenderInfo(
                73, 73, 25, 20);
        // Swapping the place of addActiveVideoUnit and addAttendeeVideoUnit would display
        // both user and expert's videos. This doesn't show the expert's video, but it does
        // stop the "Powered by Zoom" text from being displayed :)
        mobileRTCVideoViewManager.addActiveVideoUnit(renderInfo);

        Display display = ((WindowManager) getSystemService(Service.WINDOW_SERVICE))
                .getDefaultDisplay();
        int displayRotation = display.getRotation();
        inMeetingService.getInMeetingVideoController().rotateMyVideo(displayRotation);
    }

    private final InMeetingServiceListener inMeetingServiceListener = new SimpleInMeetingListener() {
        @Override
        public void onMeetingNeedCloseOtherMeeting(InMeetingEventHandler inMeetingEventHandler) {
            inMeetingEventHandler.endOtherMeeting();
        }

        @Override
        public void onMeetingFail(int errorCode, int internalErrorCode) {
            Toast.makeText(ZoomActivity.this, "Meeting Fail", Toast.LENGTH_LONG).show();
            Log.i(TAG, "onMeetingFail, errorCode=" + errorCode + ", internalErrorCode=" +
                    internalErrorCode);
        }

        @Override
        public void onMeetingLeaveComplete(long ret /*leave reason*/) {
            /* The docs mention MobileRTCVideoViewManager#destroy, but it does not exist.
            if (mobileRTCVideoViewManager != null) {
                mobileRTCVideoViewManager.destroy();
            } */
            ZoomActivity.this.finish();
        }
    };
}