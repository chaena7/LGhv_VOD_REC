import React, { useEffect } from 'react';
import 'styles/weather.css'
function Weather() {
  useEffect(() => {
    const script = document.createElement('script');
    script.src = 'https://app2.weatherwidget.org/js/?id=ww_431a5b655ffd8';
    script.async = true;

    const handleError = (error) => {
      console.error('Error loading widget script:', error);
    };

    script.addEventListener('error', handleError);

    document.body.appendChild(script);

    return () => {
      script.removeEventListener('error', handleError);
      document.body.removeChild(script);
    };
  }, []);

  return (
    <div
      id="ww_431a5b655ffd8"
      v='1.3'
      loc='id'
      a='{"t":"responsive","lang":"ko","sl_lpl":1,"ids":["wl4479"],"font":"Arial","sl_ics":"one_a","sl_sot":"celsius","cl_bkg":"#FFFFFF00","cl_font":"rgba(255,255,255,1)","cl_cloud":"#d4d4d4","cl_persp":"#2196F3","cl_sun":"#FFC107","cl_moon":"#FFC107","cl_thund":"#FF5722","sl_tof":"3","cl_odd":"#00000000","el_nme":3,"el_wfc":3}'
    >
      
      More forecasts: <a href="https://sharpweather.com/ko/seoul/" id="ww_431a5b655ffd8_u" rel="noreferrer" target="_blank">일기 예보</a>
    </div>
  );
};


export { Weather };