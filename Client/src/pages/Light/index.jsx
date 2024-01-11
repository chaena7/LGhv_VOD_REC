import React, { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Button, Img, Text, Text2, TimeComponent, Weather } from "components";
import { useParams } from 'react-router-dom';
import { CSSTransition } from "react-transition-group";
import axios from "axios"
import 'styles/loading.css';
import 'styles/animation.css';
import 'styles/smooth.css';
import 'styles/img.css';
// import LoadingScreen from "components/Loading/Loading";

function LoadingScreen() {
  return (
    <div className='lodaing-screen'>
      <img src='/images/spinner.gif' alt='로딩 중' />
    </div>
  );
};


function LightPage() {
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const { programId } = useParams();
  const [data, setData] = useState(null);
  const [data1, setData1] = useState(null);
  const [showPage, setShowPage] = useState(false);
  useEffect(() => {
    setShowPage(true);
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const access = localStorage.getItem('access_token');
        if (!access) {
          navigate('/');
          return;
        };
        setLoading(true);
        const response = await axios.get(`https://hello00back.net/vod_detail/${programId}/`);
        console.log(response);
        if (response.status === 200) {


          setData(response.data.data);
          setData1(response.data.recommend);
          console.log("관련:", response.data.recommend);
          console.log(response.data.data);
          setLoading(false);
        }
      } catch (error) {
        console.error('데이터 읽기 실패 원인:', error);
      }
    };
    fetchData();

  }, [programId, navigate]);
  const Click = (dataItem) => {
    navigate(`/Light/${dataItem}`);
  }
  const page = localStorage.getItem('page');

  let route = '/FamilyHomeLight'; // 기본 경로 설정

  if (page === '0') {
    route = '/FamilyHomeLight';
  } else if (page === '1') {
    route = '/familyhomelight/movie';
  } else if (page === '2') {
    route = '/familyhomelight/drama';
  } else if (page === '4') {
    route = '/simple';
  } else if (page === '5') {
    route = '/familyhomelight/search';
  }

  // 클릭 이벤트 핸들러
  const handleClick = () => {
    navigate(route); // 설정된 경로로 이동
  };
  const buttonStyle = {
    border: 'none',
    backgroundColor: 'transparent',
    cursor: 'pointer',
    fontSize: '24px',
    outline: 'none',
    padding: '10px',
    color: '#555',
    width: '50px', // 원하는 너비 설정
    height: '50px',
  };

  const HorizontalPosters = ({ simposter }) => {
    const containerRef = useRef(null);

    const scrollTo = (scrollOffset) => {
      const container = containerRef.current;
      container.scrollLeft += scrollOffset;
    };

    return (
      <div style={{ postion: 'relative', display: 'flex', alignItems: 'center', marginLeft: '4%', marginRight: '-562%' }}>
        <button className="button" onClick={() => scrollTo(-600)} style={{ ...buttonStyle, position: 'absolute', left: '34%' }}><img src="https://seasonmarket.co.kr/img/slider_left.png" alt="Previous" /></button>
        <div style={{ postion: 'relative', display: 'flex', overflowX: 'auto', marginLeft: '5%', marginRight: '-400%', minWidth: '78%', maxWidth: '78%' }} ref={containerRef}>
          {simposter.map((item, index) => (
            <img
              className="detail1"
              key={index}
              style={{ width: '270px', height: '330px', marginRight: '10px', border: '2px solid #ccc', borderRadius: '2px', cursor: 'pointer' }}
              src={item[2]}
              alt={`${index}`}
              onClick={() => Click(item[0])}
            />
          ))}
        </div>
        <button className="button" onClick={() => scrollTo(600)} style={{ ...buttonStyle, position: 'absolute', right: '-453%' }}>
          <img src="https://seasonmarket.co.kr/img/slider_right.png" alt="next" /></button>
      </div>
    );
  };
  return (
    <>
      <CSSTransition
        in={showPage}
        timeout={5000}
        classNames="fade"
        unmountOnExit
      >
        <div>
          <div className="bg-gray-100 border border-black-900 border-solid flex flex-col font-inter items-center justify-start mx-auto h-full w-full">
            <div className="flex flex-col items-center justify-start w-full">
              <div className="z-10 !sticky top-[0] overflow-block relative flex bg-red-A400 flex md:flex-col flex-row md:gap-5 items-start justify-end pb-1.5 px-1.5 w-full">
                <Img
                  className="button common-pointer absolute h-[45px] w-[45px] left-4 md:mt-0 mt-[26px]"
                  src="/images/img_arrowdown.svg"
                  alt="arrowdown"
                  onClick={handleClick}
                />
                <Img className="absolute font-yogi font-lighter left-[8%] mt-[1%] w-[10%] tracking-[-0.30px]"
                  src="/images/teamlogo.png">
                </Img>
                <Text className="absolute font-yogi font-lighter left-[45%] mt-[1.7%] md:text-3xl text-4xl text-white-A700 tracking-[-0.30px]">
                  DETAIL
                </Text>
                <div className="relative h-[80px] mt-[4px] mr-[1%] md:px-5 w-[20%] md:w-full">
                  <Weather />
                </div>

                <Text
                  className="ml-9 md:ml-[0] mr-[15px] md:mt-0 mt-[25px] md:text-3xl sm:text-[28px] text-[32px] text-center text-white-A700 tracking-[-0.16px]"
                  size="txtInterSemiBold32"
                >
                  <TimeComponent />
                </Text>
              </div>
              {loading ? (
                <div className='overlay'>
                  <LoadingScreen />
                </div>
              ) : (


                <div className="bg-gradient flex flex-col items-start justify-end pt-[3%] w-full">

                  <div className="flex flex-col items-center justify-start md:ml-[0] w-[95%] md:w-full">

                    <div className="flex md:flex-col flex-row md:gap-10 items-center justify-between w-full">


                      <Img
                        className="absolute ml-[8%] mt-[5%] object-cover rounded-[25px]"
                        // src={data[2]}
                        src={data ? data[2] : ""}
                        style={{ width: '450px', height: '650px', objectFit: 'cover' }}

                      />
                      <div className="flex flex-col items-start justify-start mt-[4%] ml-[40%]">
                        <Text
                          className="ml-[20%] sm:text-[40px] md:text-[46px] text-[40px] font-yogi text-white-A700 tracking-[-0.25px]"
                          size="txtYogi"
                        >
                          {data ? data[0] : ""}
                          <p><br /></p>
                        </Text>
                        <Text
                          className="ml-[20%] mt-[3%] font-yogi text-white-A700 text-2xl text-[20px] tracking-[-0.10px]"
                          size="txtYogi"
                        >
                          장르 : {data ? data[1] : ""}
                        </Text>
                        <Text
                          className="ml-[20%] leading-[30.00px] mt-[10px] text-xl md:text-[20px] text-white-A700 sm:text-xl text-[14.5px] tracking-[-0.12px]"
                          size="txtInterSemiBold10"
                        >

                          {data ? data[4] : ""}<br />{data ? data[3] : ""}

                        </Text>
                        <Text
                          className="ml-[20%] leading-[40.00px] font-yogi mt-[30px] text-2xl md:text-[22px] text-[20px] text-white-A700 sm:text-xl tracking-[-0.12px]"
                          size="txtYogi"
                        >
                          <>
                            출연
                          </>
                        </Text>
                        <Text
                          className="ml-[20%] leading-[40.00px] mt-[0px] text-xl text-[14.5px] md:text-[22px] text-white-A700 sm:text-xl tracking-[-0.12px]"
                          size="txtYogi"
                        >
                          <>
                            {data ? data[8] : ""}
                          </>
                        </Text>
                        <Text
                          className="ml-[20%] leading-[40.00px] mt-[25px] font-yogi text-2xl md:text-[22px] text-[20px] text-white-A700 sm:text-xl tracking-[-0.12px]"
                          size="txtYogi">
                          <p>줄거리 : </p>

                        </Text>
                        <Text
                          className="ml-[20%] leading-[40.00px] mt-[10px] text-xl md:text-[22px] text-[14.5px] text-white-A700 sm:text-xl tracking-[-0.12px]  mr-[2%]"
                          size="txtYogi">

                          {data ? data[7] : ""}
                        </Text>
                        <div className="flex flex-col items-start justify-start md:ml-[0] ml-[20%] mt-[5%] w-[181px]">
                          <Button
                            className="common-pointer cursor-pointer font-yogi h-[50px] text-center text-lg tracking-[-0.09px] w-[175px]"
                            // onClick={handleNavigate}
                            shape="round"
                            color="red_A400"

                            variant="fill"
                          >
                            바로 시청하기
                          </Button>
                        </div>



                      </div>

                    </div>

                  </div>
                  <Text2
                    className="sm:text-[21px] font-yogi md:text-[20px] mt-[8%]  text-white tracking-[-0.13px] w-full ml-[8%]"
                    size="txtYogi"
                  >
                    <span className="text-white text-left font-yogi text-[40px]">
                      유사한 콘텐츠{" "}
                    </span>
                    <span className="md:text-[45px] sm:text-[45px] text-red-A400 font-yogi text-left text-[45px] font-bold">
                      VOD
                    </span>
                  </Text2>

                  {/* <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  {data1.map((dataItem, index) => (
                    <div
                        key={index}
                        className="ml-auto object-cover rounded-[25px]"
                        style={{
                        width: '220px',
                        height: '300px',
                        objectFit: 'cover',
                        border: '10px', 
                        margin: '60px', 
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                      }}
                    >
                      <Img
                        src={dataItem[2]}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                        onClick={() => Click(dataItem[0])}
                      />
                  </div>
                 ))}
                </div> */}
                  <div className="flex-shrink-0 h-[250px] relative w-1/6 ml-[2%] mr-[5%] mt-[2%] mb-[15%] w-full">

                    <div className="video-container">
                      <HorizontalPosters simposter={data1} />
                    </div>

                  </div>
                  <Text
                    className="leading-[100.00px] pl-[50px] sm:text-[21px] md:text-[23px] text-[10px] text-black-900 tracking-[-0.13px] w-full"
                    style={{ backgroundColor: 'gray', width: '150%' }}
                  >
                    <span className="text-black-900 font-yogi mt-[3%]" style={{ display: 'block', marginBottom: '8px' }}>
                      Made by HELLO00 Front 황성주
                    </span>
                    <span className="text-black-900 font-yogi mt-[1%]" style={{ display: 'block', marginBottom: '8px' }}>
                      Member: 공유경, 김명현, 김은혜, 박채나, 황성주
                    </span>
                    <span className="text-black-900 font-yogi mt-[2%] mb-[3%]" style={{ display: 'block' }}>
                      Studied at LG Hello Vision DX DATA SCHOOL
                    </span>
                  </Text>
                </div>
              )}
            </div>


          </div>



        </div>

      </CSSTransition>
    </>
  )
}
export default LightPage;
