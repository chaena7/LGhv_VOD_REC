import React, { useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import 'styles/smooth.css'


const Ranklist = ({ rankposter }) => {
  const containerRef = useRef(null);
  const navigate = useNavigate();
  
  const Click = (dataItem, navigate) => {
    navigate(`/Light/${dataItem}`);
  };
  const buttonStyle = {
    border: 'none',
    backgroundColor: 'transparent',
    cursor: 'pointer',
    fontSize: '24px',
    outline: 'none',
    padding: '10px',
    color: '#555',
    width: '70px', // 원하는 너비 설정
    height: '70px',
  };
    const scrollTo = (scrollOffset) => {
      const container = containerRef.current;
      container.scrollLeft += scrollOffset;
    };

    return (
        <div style={{postion: 'relative', display:'flex', alignItems: 'center', marginLeft: '12%', marginRight: '-521%' }}>
          <button className="button" onClick={() => scrollTo(-800)} style={{...buttonStyle, position: 'absolute', left: '23%'}}><img src="https://seasonmarket.co.kr/img/slider_left.png" alt="Previous"/></button>
          <div style={{postion: 'relative', display: 'flex', overflowX: 'auto', marginLeft: '2.5%', marginRight: '0', minWidth: '96%', maxWidth: '96%'}} ref={containerRef}>
            {rankposter.map((item, index) => (
              <img
                key={index}
                style={{ width: '290px', height: '330px', marginRight: '1px', border: '2px solid #ccc', borderRadius: '5px', cursor: 'pointer' }}
                src={item[2]}
                alt={`${index}`}
                onClick={() => Click(item[0], navigate)}
              />
            ))}
          </div>
          <button className="button" onClick={() => scrollTo(800)} style={{...buttonStyle, position: 'absolute', right: '-515%'}}>
            <img src="https://seasonmarket.co.kr/img/slider_right.png" alt="next"/></button>
        </div>
      );
};

export { Ranklist };