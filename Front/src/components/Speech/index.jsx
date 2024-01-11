import React, { useState, useRef, useEffect } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import microphoneIcon from '../../assets/images/mic.svg';

const Voice = ({ onTranscript }) => {

    const commands = [
        {
            command: "open *",
            callback: (website) => {
                window.open("http://" + website.split(" ").join(""));
            },
        },
        {
            command: "reset",
            callback: () => {
                handleReset();
            },
        },
    ];
    // 수정 후 빈 객체가 삭제되었습니다.


    const { transcript, resetTranscript } = useSpeechRecognition({ commands });
    const [isListening, setIsListening] = useState(false);
    const microphoneRef = useRef(null);
    useEffect(() => {
        if (transcript) {
            onTranscript(transcript); // onTranscript 함수에 최종 transcript를 전달

        }
    }, [onTranscript, transcript]);

    if (!SpeechRecognition.browserSupportsSpeechRecognition()) {
        return (
            <div className='microphone-container'>
                Browser is not Support Speech Recognition.
            </div>
        );
    }

    const handleListening = () => {
        if (isListening === false) {
            resetTranscript();

            setIsListening(true);
            microphoneRef.current.classList.add('listening');
            SpeechRecognition.startListening({
                continuous: true,
                language: 'ko'
            });
        } else {
            setIsListening(false);
            microphoneRef.current.classList.remove('listening');
            SpeechRecognition.stopListening({
                continuous: false,
            });

        }
    };

    const handleReset = () => {
        setIsListening(false);
        microphoneRef.current.classList.remove('listening');
        SpeechRecognition.stopListening();
        resetTranscript();
    };

    console.log(transcript)




    return (
        <div className='microphone-wrapper'>
            <div className='microphone-container'>

                <div
                    className='microphone-icon-container'
                    ref={microphoneRef}
                    onClick={handleListening}
                    style={{
                        animation: isListening ? 'pulse 0.5s infinite' : 'none'
                    }}
                >

                    <img src={microphoneIcon} className='microphone-icon' alt='icon' />

                </div>
                <div className='microphone-status'>
                    {isListening ? "음성 입력 중입니다" : ""}
                    {isListening !== true ? "음성 입력 혹은 수정은 클릭!" : ""}
                </div>


            </div>
        </div>
    );
};

export { Voice };