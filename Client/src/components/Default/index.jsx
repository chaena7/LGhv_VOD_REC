import React from 'react';
import { useNavigate } from 'react-router-dom';

function Default({ poster }) {
    const navigate = useNavigate();
    
    const Click = (dataItem, navigate) => {
        navigate(`/Light/${dataItem}`);
    
    };

    return (
        <div style={{ position: 'relative', display: 'flex', alignItems: 'center', marginRight: '0%', marginTop: '2%', width: '100%' }}>
        <div style={{ position: 'relative', display: 'flex', flexWrap: 'wrap', height: '333px', marginRight: '0%', minWidth: '100%', maxWidth: '100%'}}>
            {(poster.slice(0, 7)).map((item, index) => (
                <div key={index} style={{ width: '12%', 
                minWidth: '12%', 
                maxWidth: '12%', 
                flex: '0 0 auto', 
                marginRight: (index + 1) % 7 !== 0 && index !== poster.length - 1 ? '1.6%' : '0', 
                marginBottom: '24px', 
                }}>
                    <img
                        className='img'
                        style={{ width: '200px', height: '280px', boxShadow: '0 0 10px rgba(0, 0, 0, 0.5)', 
                        border: '1px solid rgba(0, 0, 0, 0.2)', cursor: 'pointer' }}
                        src={item[2]}
                        alt={`${index}`}
                        onClick={() => Click(item[0], navigate)}
                    />
                </div>
            ))}
        </div>
    </div>
    );
}

export { Default };