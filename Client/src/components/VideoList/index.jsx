import { useNavigate } from "react-router-dom";
import 'styles/img.css'
const VideoItem = ({ poster, title, genre, actor, summary, onClick }) => {
  return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          width: '100%',
          border: '3px',
          borderRadius: '15px',
          boxShadow: '0 0 10px 5px rgba(201, 213, 214, 0.4)',
        }}
        onClick={onClick}
      >
      <img src={poster} alt={title} style={{ width:'15%', borderRadius: '10px'}} />
      <div className="w-full">
      <div className="font-yogi text-[200%] w-[100%] ml-[10%]">
        {title}
      </div>
      <div className="font-yogi mt-[5%] text-[110%] ml-[10%]">
        {genre}
      </div>
      <div className="font-yogi ml-[10%] mt-[3%] w-[70%]">
        출연: {actor}
      </div>
      <div className="font-yogi ml-[10%] mt-[3%] w-[75%]">
        줄거리: {summary}
      </div>
      </div>
    </div>
  );
};

const VideoList = ({ data }) => {
  const navigate = useNavigate();
  const filteredData = data ? data.filter(item => item !== null) : [];
  
  const handleClick = (dataItem) => {
    navigate(`/Light/${dataItem}`);
  };

  return (
    <div>
      {filteredData.map((item, index) => {
        const combinedGenre = `${item[1]}, ${item[4]}`;
        return (
          <div className="button2 w-full mt-[3%]">
          <VideoItem
            key={index}
            poster={item[2]}
            title={item[0]}
            genre={combinedGenre}
            actor={item[8]}
            summary={item[7]}
            onClick={() => handleClick(item[6])}
          />
          </div>
        );
      })}
    </div>
  );


   


};

export { VideoList };


