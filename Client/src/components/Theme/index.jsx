import React, { useState } from 'react';
import 'styles/circle.css';
function Theme() {


    const [animate, setAnimate] = useState(false);

    const animateShine = () => {
        setAnimate(prevAnimate => !prevAnimate);
 // 1초 후에 애니메이션 종료
    };

    return (
        <div className={animate ? 'circle move-right' : 'circle'} onClick={animateShine}>
            <div className={animate ? 'shine animate' : 'shine'}></div>
        </div>
    );
    // return (
    //     <div className={moveRight ? 'circle move-right' : 'circle'} onClick={animateShine}>
    //       <div className="shine"></div>
    //     </div>
    //   );
};

export { Theme };