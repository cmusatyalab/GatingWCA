ZoomMtg.preLoadWasm();
ZoomMtg.prepareJssdk();

ZoomMtg.init({
    debug: false,

    // Reload the Zoom page if the the user
    // leaves meeting.
    leaveUrl: '/zoom',

    showMeetingHeader: false,
    disableInvite: true,
    videoHeader: false,
    isSupportAV: true,
    success: function() {
	ZoomMtg.join({
	    signature: signature,
	    apiKey: apiKey,
	    meetingNumber: meetingNumber,
	    userName: userName,
	    passWord: passWord,
	    error: function (res) {
		console.log(res);
	    },
	});
    }
});
