import React, { useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import "styles/smooth.css"
import "styles/list.css"

const Reclist = ({ rankposter }) => {
  const containerRef = useRef(null);
  const navigate = useNavigate();
  
  const Click = (dataItem, navigate) => {
    navigate(`/Light/${dataItem}`);
  };

  const scrollTo = (scrollOffset) => {
    const container = containerRef.current;
    container.scrollLeft += scrollOffset;
  };

  return (
    <div style={{ position: 'relative', display: 'flex', alignItems: 'center', marginLeft: '11.5%', marginRight: '-500%', marginTop: '2.5%' }}>
      <button className="button" onClick={() => scrollTo(-600)} style={{ width: '45px', position: 'absolute', left: '1.5%', zIndex: '1'  }}>
        <img src="https://seasonmarket.co.kr/img/slider_left.png" alt="Previous" /></button>
      <div style={{ position: 'relative', display: 'flex', overflowX: 'auto', height:'333px', marginLeft: '2%', marginRight: '0', minWidth: '100%', maxWidth: '100%' }} ref={containerRef}>
        {rankposter.map((item, index) => (
          <img className='img'
            key={index}
            style={{ width: '260px', height: '320px', marginRight: '12px', border:'1px' , cursor: 'pointer' }}
            src={item[2]}
            alt={`${index}`}
            onClick={() => Click(item[0], navigate)}
          />
        ))}
      </div>
      <button className="button" onClick={() => scrollTo(600)} style={{ width: '45px', position: 'absolute', right: '-2.5%' }}>
        <img src="https://seasonmarket.co.kr/img/slider_right.png" alt="next" />
      </button>
    </div>
  );
};

export { Reclist };
