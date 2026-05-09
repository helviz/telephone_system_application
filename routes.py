from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

app = Flask(__name__)

@app.route("/twilio-voice", methods=['POST'])
def twilio_voice():
    """Returns TwiML to initiate a Twilio Media Stream."""
    resp = VoiceResponse()
    connect = Connect()
    # Replace with your actual public wss:// URL (e.g., ngrok)
    connect.stream(url=f'wss://{request.host}/media-stream/twilio')
    resp.append(connect)
    return str(resp)

@app.route("/telnyx-voice", methods=['POST'])
def telnyx_voice():
    """Returns TeXML to initiate a Telnyx Call Control Stream."""
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Stream url="wss://{request.host}/media-stream/telnyx" />
    </Response>"""
    return Response(texml, mimetype='text/xml')