package edu.cmu.cs.wca;

import java.util.List;

import us.zoom.sdk.ChatMessageDeleteType;
import us.zoom.sdk.FreeMeetingNeedUpgradeType;
import us.zoom.sdk.IRequestLocalRecordingPrivilegeHandler;
import us.zoom.sdk.InMeetingAudioController;
import us.zoom.sdk.InMeetingChatController;
import us.zoom.sdk.InMeetingChatMessage;
import us.zoom.sdk.InMeetingEventHandler;
import us.zoom.sdk.InMeetingServiceListener;
import us.zoom.sdk.LocalRecordingRequestPrivilegeStatus;
import us.zoom.sdk.MobileRTCFocusModeShareType;
import us.zoom.sdk.VideoQuality;

// Taken from:
// mobilertc-android-studio/sample/src/main/java/us/zoom/sdksample/inmeetingfunction/customizedmeetingui/SimpleInMeetingListener.java

public abstract class SimpleInMeetingListener implements InMeetingServiceListener {

    @Override
    public void onMeetingNeedPasswordOrDisplayName(boolean b, boolean b1, InMeetingEventHandler inMeetingEventHandler) {

    }

    @Override
    public void onWebinarNeedRegister(String registerUrl) {

    }

    @Override
    public void onJoinWebinarNeedUserNameAndEmail(InMeetingEventHandler inMeetingEventHandler) {

    }

    @Override
    public void onMeetingNeedCloseOtherMeeting(InMeetingEventHandler inMeetingEventHandler) {

    }

    @Override
    public void onMeetingFail(int i, int i1) {

    }

    @Override
    public void onMeetingLeaveComplete(long l) {

    }

    @Override
    public void onMeetingUserJoin(List<Long> list) {

    }

    @Override
    public void onMeetingUserLeave(List<Long> list) {

    }

    @Override
    public void onMeetingUserUpdated(long l) {

    }

    @Override
    public void onMeetingHostChanged(long l) {

    }

    @Override
    public void onMeetingCoHostChanged(long l) {

    }

    @Override
    public void onActiveVideoUserChanged(long var1) {

    }

    @Override
    public void onActiveSpeakerVideoUserChanged(long var1) {

    }

    @Override
    public void onSpotlightVideoChanged(boolean b) {

    }

    @Override
    public void onUserVideoStatusChanged(long userId, VideoStatus status) {

    }

    @Override
    public void onMicrophoneStatusError(InMeetingAudioController.MobileRTCMicrophoneError mobileRTCMicrophoneError) {

    }

    @Override
    public void onUserAudioStatusChanged(long userId, AudioStatus audioStatus) {

    }

    @Override
    public void onUserAudioTypeChanged(long l) {

    }

    @Override
    public void onMyAudioSourceTypeChanged(int i) {

    }

    @Override
    public void onLowOrRaiseHandStatusChanged(long l, boolean b) {

    }

    @Override
    public void onChatMessageReceived(InMeetingChatMessage inMeetingChatMessage) {

    }

    @Override
    public void onUserNetworkQualityChanged(long userId) {


    }

    @Override
    public void onHostAskUnMute(long userId) {

    }

    @Override
    public void onHostAskStartVideo(long userId) {

    }

    @Override
    public void onSilentModeChanged(boolean inSilentMode){

    }

    @Override
    public void onFreeMeetingReminder(boolean isOrignalHost, boolean canUpgrade, boolean isFirstGift){

    }

    @Override
    public void onMeetingActiveVideo(long userId) {

    }

    @Override
    public void onSinkAttendeeChatPriviledgeChanged(int privilege) {

    }

    @Override
    public void onSinkAllowAttendeeChatNotification(int privilege) {

    }

    @Override
    public void onUserNameChanged(long userId, String name) {

    }

    @Override
    public void onFreeMeetingNeedToUpgrade(FreeMeetingNeedUpgradeType type, String gifUrl) {

    }

    @Override
    public void onFreeMeetingUpgradeToGiftFreeTrialStart() {

    }

    @Override
    public void onFreeMeetingUpgradeToGiftFreeTrialStop() {

    }

    @Override
    public void onFreeMeetingUpgradeToProMeeting() {

    }

    @Override
    public void onRecordingStatus(RecordingStatus status) {

    }

    @Override
    public void onInvalidReclaimHostkey() {

    }

    @Override
    public void onInMeetingUserAvatarPathUpdated(long l) {

    }

    @Override
    public void onMeetingCoHostChange(long l, boolean b) {

    }

    @Override
    public void onHostVideoOrderUpdated(List<Long> list) {

    }

    @Override
    public void onFollowHostVideoOrderChanged(boolean b) {

    }

    @Override
    public void onSpotlightVideoChanged(List<Long> list) {

    }

    @Override
    public void onSinkMeetingVideoQualityChanged(VideoQuality videoQuality, long l) {

    }

    @Override
    public void onChatMsgDeleteNotification(String s, ChatMessageDeleteType chatMessageDeleteType) {

    }

    @Override
    public void onShareMeetingChatStatusChanged(boolean b) {

    }

    @Override
    public void onSinkPanelistChatPrivilegeChanged(InMeetingChatController.MobileRTCWebinarPanelistChatPrivilege mobileRTCWebinarPanelistChatPrivilege) {

    }

    @Override
    public void onUserNamesChanged(List<Long> list) {

    }

    @Override
    public void onClosedCaptionReceived(String s, long l) {

    }

    @Override
    public void onLocalRecordingStatus(long l, RecordingStatus recordingStatus) {

    }

    @Override
    public void onPermissionRequested(String[] strings) {

    }

    @Override
    public void onAllHandsLowered() {

    }

    @Override
    public void onLocalVideoOrderUpdated(List<Long> list) {

    }

    @Override
    public void onLocalRecordingPrivilegeRequested(IRequestLocalRecordingPrivilegeHandler iRequestLocalRecordingPrivilegeHandler) {

    }

    @Override
    public void onSuspendParticipantsActivities() {

    }

    @Override
    public void onAllowParticipantsStartVideoNotification(boolean b) {

    }

    @Override
    public void onAllowParticipantsRenameNotification(boolean b) {

    }

    @Override
    public void onAllowParticipantsUnmuteSelfNotification(boolean b) {

    }

    @Override
    public void onAllowParticipantsShareWhiteBoardNotification(boolean b) {

    }

    @Override
    public void onMeetingLockStatus(boolean b) {

    }

    @Override
    public void onRequestLocalRecordingPrivilegeChanged(LocalRecordingRequestPrivilegeStatus localRecordingRequestPrivilegeStatus) {

    }

    @Override
    public void onAICompanionActiveChangeNotice(boolean b) {

    }

    @Override
    public void onParticipantProfilePictureStatusChange(boolean b) {

    }

    @Override
    public void onCloudRecordingStorageFull(long l) {

    }

    @Override
    public void onUVCCameraStatusChange(String s, UVCCameraStatus uvcCameraStatus) {

    }

    @Override
    public void onFocusModeStateChanged(boolean b) {

    }

    @Override
    public void onFocusModeShareTypeChanged(MobileRTCFocusModeShareType mobileRTCFocusModeShareType) {

    }

    @Override
    public void onVideoAlphaChannelStatusChanged(boolean b) {

    }

    @Override
    public void onAllowParticipantsRequestCloudRecording(boolean b) {

    }
}