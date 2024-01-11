import React, { useState, useEffect } from 'react';

const TimeComponent = () => {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000); // 1초마다 갱신

    return () => clearInterval(interval);
  }, []);

  const formattedTime = `${currentTime.getHours()}:${('0' + currentTime.getMinutes()).slice(-2)}`; // 시:분 형식으로 포맷

  return (
    <div>
      {formattedTime}
    </div>
  );
};

export { TimeComponent };
