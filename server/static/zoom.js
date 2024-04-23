window.addEventListener("DOMContentLoaded", function (event) {
  console.log("DOM fully loaded and parsed");
  websdkready();
});

function websdkready() {
  var meetingConfig = {
    meetingNumber: meetingNumber,
    userName: userName,
    userEmail: userEmail,
    passWord: passWord,
    sdkKey: clientID,
    signature: jwtToken,
    // Reload the Zoom page if the user leaves meeting
    leaveUrl: "/zoom"
  };
  console.log(JSON.stringify(ZoomMtg.checkSystemRequirements()));
  ZoomMtg.preLoadWasm();
  ZoomMtg.prepareWebSDK();

  function beginJoin(signature) {
    ZoomMtg.i18n.load(meetingConfig.lang);
    ZoomMtg.init({
      leaveUrl: meetingConfig.leaveUrl,
      webEndpoint: meetingConfig.webEndpoint,
      disableCORP: !window.crossOriginIsolated, // default true
      // disablePreview: false, // default false
      externalLinkPage: "./externalLinkPage.html",
      success: function () {
        console.log(meetingConfig);
        console.log("signature", signature);

        ZoomMtg.join({
          meetingNumber: meetingConfig.meetingNumber,
          userName: meetingConfig.userName,
          signature: signature,
          sdkKey: meetingConfig.sdkKey,
          userEmail: meetingConfig.userEmail,
          passWord: meetingConfig.passWord,
          success: function (res) {
            console.log("join meeting success");
            console.log("get attendeelist");
            ZoomMtg.getAttendeeslist({});
            ZoomMtg.getCurrentUser({
              success: function (res) {
                console.log("success getCurrentUser", res.result.currentUser);
              },
            });
          },
          error: function (res) {
            console.log(res);
          },
        });
      },
      error: function (res) {
        console.log(res);
      },
    });

    ZoomMtg.inMeetingServiceListener("onUserJoin", function (data) {
      console.log("inMeetingServiceListener onUserJoin", data);
    });

    ZoomMtg.inMeetingServiceListener("onUserLeave", function (data) {
      console.log("inMeetingServiceListener onUserLeave", data);
    });

    ZoomMtg.inMeetingServiceListener("onUserIsInWaitingRoom", function (data) {
      console.log("inMeetingServiceListener onUserIsInWaitingRoom", data);
    });

    ZoomMtg.inMeetingServiceListener("onMeetingStatus", function (data) {
      console.log("inMeetingServiceListener onMeetingStatus", data);
    });
  }

  beginJoin(meetingConfig.signature);
}
