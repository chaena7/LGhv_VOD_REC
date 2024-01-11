import React, { useState, useEffect } from "react";


import 'styles/banner.css'; 

const Firstlist = () => { 
    const imagenames = ['images/shfid.png', 'images/spider.jpg','images/seoul1.jpg', '/images/sweet2.jpg', '/images/boygen.jpg', '/images/lucky.jpg', '/images/nowdie.png'] 
    const [currentImageIndex, setCurrentImageIndex] = useState(0);
    useEffect(() => {
        const interval = setInterval(() => {
          setCurrentImageIndex((prevIndex) => (prevIndex + 1) % imagenames.length);
        }, 7000);
    
        return () => clearInterval(interval);
      }, [imagenames.length]);
      
      const handlePrevClick = () => {
        setCurrentImageIndex((prevIndex) => (prevIndex - 1 + imagenames.length) % imagenames.length);
      };
    
      const handleNextClick = () => {
        setCurrentImageIndex((prevIndex) => (prevIndex + 1) % imagenames.length);
      };
      const leftImageIndex = (currentImageIndex - 1 + imagenames.length) % imagenames.length;
      const rightImageIndex = (currentImageIndex + 1) % imagenames.length;
      const currentindex = currentImageIndex;
      return(
        <>



                <div className="image-slider-container mt-[1%]" style={{
                width: '92%',
                height: '570px',
                position: 'relative',
                overflow: 'visible',
                marginLeft: '3.5%'
                }}>

                <div className="left-half" style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '50%',
                height: '100%',
                background: `linear-gradient(to bottom, rgb(0, 0,0, 0.8) 0%, rgba(0, 0, 0, 0.5) 50%, rgba(0, 0, 0, 0.8) 100%), url(${imagenames[leftImageIndex]})`,

                }}></div>
                
                <div className="right-half" style={{
                position: 'absolute',
                top: 0,
                right: 0,
                width: '50%',
                height: '100%',
                background: `linear-gradient(to bottom, rgb(0, 0,0, 0.8) 0%, rgba(0, 0, 0, 0.5) 50%, rgba(0, 0, 0, 0.8) 100%), url(${imagenames[rightImageIndex]})`,

                }}></div>

                <div className="absolute button-wrapper">
                  <button className="left-button" onClick={handlePrevClick}>
                    <img src="https://seasonmarket.co.kr/img/slider_left.png" alt="Previous"/>
                  </button>
                  </div>
                <div className="image-slider">
                
                

                {imagenames.map((image, index) => (
                  <img
                    style={{ border: '7px solid #e9e7e7', borderRadius: '12px' }}
                    key={index}
                    src={image}
                    alt={`Slide ${index + 1}`}
                    className={`${index === currentindex ? 'active' : ''} border-radius`}
                    
                  />
                  ))}
          </div>

                <div className="absolute button-wrapper right-0"> 
                  <button className="right-button" onClick={handleNextClick}>
                    <img src="https://seasonmarket.co.kr/img/slider_right.png" alt="Next"/>
                  </button>

                </div>

            {/* <div className="relative image-slider-wrapper"> */}


            </div>

        </>
      )

};

export { Firstlist }