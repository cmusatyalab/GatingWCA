let oldState = null;
const socket = new WebSocket('wss://' + location.host + '/ws');

socket.addEventListener('message', function (event) {
    var msg = JSON.parse(event.data);

    if (msg.zoom_action === 'start') {
        oldState = $('#' + msg.step)[0];
        oldState.style.borderColor = 'blue';
        $('#nothelping').hide()
        $('#states').show()
    } else if (msg.zoom_action === 'stop') {
        oldState.style.removeProperty('border-color');
        $('#nothelping').show()
        $('#states').hide()
    }
});

$(document).ready(function() {
    $('.state').click(function() {
        if (oldState === this) {
            return;
        }

        if (oldState !== null) {
            oldState.style.removeProperty('border-color');
        }
        this.style.borderColor = 'blue';
        oldState = this;

        // Send this.id to client
        var msg = {
            step: this.id,
        };
        socket.send(JSON.stringify(msg));
    });
});
