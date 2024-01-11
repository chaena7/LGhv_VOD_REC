import React, { useState, useEffect } from 'react';
import 'styles/animation.css'
import { Sidebar } from "react-pro-sidebar";
import { CSSTransition } from 'react-transition-group';
import { useNavigate } from "react-router-dom";
import axios from 'axios';
import {
  Img, Text, TimeComponent, Reclist, Firstlist, Drama,
  Mov
} from "components";
import 'styles/loading.css';
import 'styles/font.css';
import 'styles/img.css';
import 'styles/clicked.css';
import 'styles/scroll.css';
import 'styles/rerec.css';
import 'styles/bar.css';






const LoadingScreen = () => {
  return (
    <div className='lodaing-screen'>

      <img src='/images/spinner.gif' alt='로딩 중' />
    </div>
  );
};


function FamilyHomeLightPage() {
  const [loading, setLoading] = useState(true);
  const [genposter, setGenposter] = useState([]);
  const [rankposter, setRankposter] = useState([]);
  const [recposter, setRecposter] = useState([]);
  const [yetposter, setYetposter] = useState([]);
  const [current, setCurrent] = useState('');



  localStorage.setItem('name', 0)
  const initialStartIndex = localStorage.getItem('startIndex');
  const initialIndex = initialStartIndex !== null ? parseInt(initialStartIndex) : 0;

  const initialStartIndex1 = localStorage.getItem('startIndex1');
  const initialIndex1 = initialStartIndex1 !== null ? parseInt(initialStartIndex1) : 0;

  const initialStartIndex2 = localStorage.getItem('startIndex2');
  const initialIndex2 = initialStartIndex2 !== null ? parseInt(initialStartIndex2) : 0;

  const initialStartIndex3 = localStorage.getItem('startIndex3');
  const initialIndex3 = initialStartIndex3 !== null ? parseInt(initialStartIndex3) : 0;

  const [startIndex, setStartIndex] = useState(initialIndex);
  const [startIndex1, setStartIndex1] = useState(initialIndex1);
  const [startIndex2, setStartIndex2] = useState(initialIndex2);
  const [startIndex3, setStartIndex3] = useState(initialIndex3);

  const [userposter, setUserposter] = useState([]);
  const [showPage, setShowPage] = useState(false);
  const [recposter1, setRecposter1] = useState([]);
  const [rankposter1, setRankposter1] = useState([]);
  const [ctcl1, setCtcl1] = useState([]);
  const [ctcl2, setCtcl2] = useState([]);
  const [ctcl3, setCtcl3] = useState([]);
  const [ctcl4, setCtcl4] = useState([]);
  const [ctcl5, setCtcl5] = useState([]);
  const [ctcl6, setCtcl6] = useState([]);
  const [ctclname, setCtclname] = useState([]);

  const [data, setData] = useState([]);
  const [data1, setData1] = useState([]);


  // 스크롤 위치
  // const [homeButton, setHome] = useState('');
  // const [dramaButton, setDrama] = useState('');
  // const [movieButton, setMovie] = useState('');
  // const [topButton, setTop] = useState('');


  // const [recSection, setRecsection] = useState('');
  // const [dramaSection, setDramasection] = useState('');
  // const [movieSection, setMoviesection] = useState('');
  // const [endSection, setEndsection] = useState('');


  //영화 데이터
  const [rankposter11, setRankposter11] = useState([]);
  const [ctcl11, setCtcl11] = useState([]);
  const [ctcl21, setCtcl21] = useState([]);
  const [ctcl31, setCtcl31] = useState([]);
  const [ctcl41, setCtcl41] = useState([]);
  const [ctcl51, setCtcl51] = useState([]);
  const [ctcl61, setCtcl61] = useState([]);
  const [ctcl71, setCtcl71] = useState([]);
  const [ctclname1, setCtclname1] = useState([]);
  localStorage.setItem('page', 0); // 기본
  const navigate = useNavigate();
  const access = localStorage.getItem('access_token');

  useEffect(() => {
    setShowPage(true);
  }, []);

  useEffect(() => {



    const fetchData = async () => {
      try {
        const access = localStorage.getItem('access_token');
        if (!access) {
          navigate('/');
          return; // 로그인 페이지로 이동 후 함수 종료
        }



      } catch (error) {
        let loadingTimer = setTimeout(() => {
          setLoading(false); // 10초 후에 로딩 완료
          // 실패 로직 필요시 추가
          clearTimeout(loadingTimer); // 타이머 초기화
        }, 10000);
        console.error('Error fetching data:', error);
      }



      try {
        const access = localStorage.getItem('access_token');
        if (!access) {
          navigate('/');
          return; // 로그인 페이지로 이동 후 함수 종료
        }

        // console.log('access token 은', access);
        setLoading(true);

        const gen = JSON.parse(localStorage.getItem('genre'));
        const genre = gen ? gen.join(',') : null;
        // console.log(genre);



        const response = await axios.get('https://hello00back.net/vodrec', {
          headers: {
            Authorization: access,
            Data: genre,
          },

        });
        //드라마
        const response1 = await axios.get('https://hello00back.net/home/drama', {
          headers: {
            Authorization: access,
          },
        });
        //영화
        const response2 = await axios.get('https://hello00back.net/home/movie', {
          headers: {
            Authorization: access,
          },



        });

        if (response.status === 200 && response1.status === 200 && response2.status === 200) {
          // console.log("여긴 문제가 아니야")
          setData(response.data.data);
          setData1(response1.data.data);

          const data = response.data.data;
          const current = response.data.current;
          // console.log(current);

          const rankposter = data[1].map(item => item);

          const yetItems = data[4]


          // console.log(data);

          // const rankposter = rankItems.map(item => item); //주간 랭킹



          const yetposter = yetItems.map(item => item); //덜 본거
          setRankposter(rankposter);

          setYetposter(yetposter);
          setCurrent(current);
          setLoading(false);


          //드라마 데이터 받아오기
          const data1 = response1.data.data; //포스터 데이터
          const rankItems1 = data1[0];

          const ctcl1Items = data1[2][0];
          const ctcl2Items = data1[2][1];
          const ctcl3Items = data1[2][2];
          const ctcl4Items = data1[2][3];
          const ctcl5Items = data1[2][4];
          const ctcl6Items = data1[2][5];
          const ctclnames = response1.data.genres; //장르 데이터

          const rankposter1 = rankItems1.map(item => item);

          const ctcl1 = ctcl1Items.map(item => item);
          const ctcl2 = ctcl2Items.map(item => item);
          const ctcl3 = ctcl3Items.map(item => item);
          const ctcl4 = ctcl4Items.map(item => item);
          const ctcl5 = ctcl5Items.map(item => item);
          const ctcl6 = ctcl6Items.map(item => item);
          const ctclname = ctclnames;

          setRankposter1(rankposter1);
          setCtcl1(ctcl1);
          setCtcl2(ctcl2);
          setCtcl3(ctcl3);
          setCtcl4(ctcl4);
          setCtcl5(ctcl5);
          setCtcl6(ctcl6);

          setCtclname(ctclname);

          //영화 데이터
          const data2 = response2.data.data; //포스터 데이터
          const rankItems2 = data2[0];
          const ctcl1Items1 = data2[1][0];
          const ctcl2Items1 = data2[1][1];
          const ctcl3Items1 = data2[1][2];
          const ctcl4Items1 = data2[1][3];
          const ctcl5Items1 = data2[1][4];
          const ctcl6Items1 = data2[1][5];
          const ctcl7Items1 = data2[1][6];
          // console.log(response2.data);
          const ctclnames1 = response2.data.genres; //장르 데이터



          const rankposter11 = rankItems2.map(item => item);
          localStorage.setItem('rank1', JSON.stringify(rankposter));
          localStorage.setItem('rank2', JSON.stringify(rankposter1));
          localStorage.setItem('rank3', JSON.stringify(rankposter11));

          const ctcl11 = ctcl1Items1.map(item => item);
          const ctcl21 = ctcl2Items1.map(item => item);
          const ctcl31 = ctcl3Items1.map(item => item);
          const ctcl41 = ctcl4Items1.map(item => item);
          const ctcl51 = ctcl5Items1.map(item => item);
          const ctcl61 = ctcl6Items1.map(item => item);
          const ctcl71 = ctcl7Items1.map(item => item);
          const ctclname1 = ctclnames1;
          console.log(ctclnames);
          setRankposter11(rankposter11);
          setCtcl11(ctcl11);
          setCtcl21(ctcl21);
          setCtcl31(ctcl31);
          setCtcl41(ctcl41);
          setCtcl51(ctcl51);
          setCtcl61(ctcl61);
          setCtcl71(ctcl71);
          setCtclname1(ctclname1);






        }


      } catch (error) {
        let loadingTimer = setTimeout(() => {
          setLoading(false); // 10초 후에 로딩 완료
          // 실패 로직 필요시 추가
          clearTimeout(loadingTimer); // 타이머 초기화
        }, 10000);
        console.error('Error fetching:', error);
      }


    };
    window.scrollTo({ top: 0, behavior: 'smooth' })
    fetchData();
  }, [navigate]);






  const handleClick = () => {
    return (
      window.scrollTo({ top: 0, behavior: 'smooth' })
    )
  };
  const handleClick1 = () => {
    return (
      window.scrollTo({ top: 0, behavior: 'smooth' })
    )
  };

  //첫번째 startIndex 설정
  useEffect(() => {
    const gen = JSON.parse(localStorage.getItem('genre'));
    const genre = gen ? gen.join(',') : null;


    const getdata = async () => {
      const result = await axios.get('https://hello00back.net/vodrec', {
        headers: {
          Authorization: access,
          Data: genre,
        },

      });
      if (result.status === 200) {
        const data11 = result.data.data;
        const genposter = (data11[0].slice(startIndex, startIndex + 10)).map(item => item);
        setGenposter(genposter);

      }
    }

    getdata();
  }, [access, startIndex]);


  //두번째 데이터
  useEffect(() => {
    const gen = JSON.parse(localStorage.getItem('genre'));
    const genre = gen ? gen.join(',') : null;
    console.log(genre);

    const getdata1 = async () => {
      try {
        const result = await axios.get('https://hello00back.net/vodrec', {
          headers: {
            Authorization: access,
            Data: genre,
          },

        });
        if (result.status === 200) {
          const data11 = result.data.data;
          const userposter = (data11[2].slice(startIndex1, startIndex1 + 10)).map(item => item)
          setUserposter(userposter);
        }
      } catch (error) {
        console.log(error);
      }
    }
    getdata1();
  }, [access, startIndex1]); //userposter 데이터 정의


  useEffect(() => {
    const gen = JSON.parse(localStorage.getItem('genre'));
    const genre = gen ? gen.join(',') : null;
    console.log(genre);

    const getdata2 = async () => {
      try {
        const result = await axios.get('https://hello00back.net/vodrec', {
          headers: {
            Authorization: access,
            Data: genre,
          },

        });
        if (result.status === 200) {
          const data11 = result.data.data;
          const recposter = (data11[3].slice(startIndex2, startIndex2 + 10)).map(item => item)
          setRecposter(recposter);

        }
      } catch (error) {
        console.log(error);
      }
    }
    getdata2();
  }, [access, startIndex2]); //recposter 데이터 정의


  useEffect(() => {
    const getdata3 = async () => {
      try {
        const result = await axios.get('https://hello00back.net/home/drama', {
          headers: {

            Authorization: access,
          },

        });
        if (result.status === 200) {
          const startIndex3 = localStorage.getItem('startIndex3');
          const data1 = result.data.data;
          const recposter1 = (data1[1].slice(startIndex3, startIndex3 + 10)).map(item => item);
          setRecposter1(recposter1);

        }
      } catch (error) {
        console.log(error);
      }
    }
    getdata3();
  }, [access, startIndex3]);


  const handleRerec = (dataArr, setStarrtIndex, setPoster, localStorageKey) => {

    if (startIndex < 90) {
      const currentIndex = startIndex + 10;
      setStarrtIndex(currentIndex);
      const items = dataArr.slice(currentIndex, currentIndex + 10).map(item => item);
      setPoster(items);
      localStorage.setItem(localStorageKey, currentIndex);
    } else {
      setStarrtIndex(0);
      const items = dataArr.slice(0, 10).map(item => item);
      setPoster(items);
      localStorage.setItem(localStorageKey, 0);
    }
  };

  const Rerec = () => {
    handleRerec(data[0], setStartIndex, setGenposter, 'startIndex');
  };



  const Rerec1 = () => {
    handleRerec(data[2], setStartIndex1, setUserposter, 'startIndex1');
  };

  const Rerec2 = () => {
    handleRerec(data[3], setStartIndex2, setRecposter, 'startIndex2');
  };

  const Rerec3 = () => {
    handleRerec(data1[1], setStartIndex3, setRecposter1, 'startIndex3');
  };




  const homeButton = document.getElementById('home-button');
  const dramaButton = document.getElementById('drama-button');
  const movieButton = document.getElementById('movie-button');
  const topButton = document.getElementById('top-button');
  const topButton1 = document.getElementById('totop');


  const recSection = document.getElementById('rec-page');
  const dramaSection = document.getElementById('drama-page');
  const movieSection = document.getElementById('movie-page');
  const endSection = document.getElementById('end-page');



  const handleScroll = () => {
    const scrollPosition = window.scrollY;
    if (recSection && dramaSection && movieSection && endSection) {
      if (scrollPosition > 100) {
        topButton1.classList.add('visible');
      } else {
        topButton1.classList.remove('visible');
      }
      if (scrollPosition >= 0 && scrollPosition < recSection.offsetTop - 100) {
        topButton.classList.add('scroll-background');
      } else {
        topButton.classList.remove('scroll-background');
      }

      if (scrollPosition >= recSection.offsetTop - 10 && scrollPosition < dramaSection.offsetTop - 100) {
        homeButton.classList.add('scroll-background');
      } else {
        homeButton.classList.remove('scroll-background');
      }

      if (scrollPosition >= dramaSection.offsetTop - 100 && scrollPosition < movieSection.offsetTop - 100) {
        dramaButton.classList.add('scroll-background');
      } else {
        dramaButton.classList.remove('scroll-background');
      }

      if (scrollPosition >= movieSection.offsetTop - 100 && scrollPosition <= endSection.offsetTop) {
        movieButton.classList.add('scroll-background');
      } else {
        movieButton.classList.remove('scroll-background');
      }

    }
  };
  window.addEventListener('scroll', handleScroll);

  const scrollToRec = () => {

    recSection.scrollIntoView({ behavior: 'smooth' });
    console.log('recSection:', recSection);
  };

  const scrollToDrama = () => {
    // const dramaSection = document.getElementById('drama-page');
    dramaSection.scrollIntoView({ behavior: 'smooth' });
  };
  const scrollToMovie = () => {
    // const movieSection = document.getElementById('movie-page');
    movieSection.scrollIntoView({ behavior: 'smooth' });
  };
  const redirectToSeasonMarket = () => {
    window.location.href = 'https://seasonmarket.co.kr/';
  };









  return (
    <>
      <CSSTransition
        in={showPage}
        timeout={1000}
        classNames="fade"
        unmountOnExit
      >
        <div>

          <div className="bg-gray-100 border border-black-900 border-solid flex flex-col font-inter items-center justify-start mx-auto w-full">

            <div className="flex flex-col items-center justify-start w-full">

              <div className="z-10 !sticky top-[0] !h-[90px]  overflow-block relative flex bg-red-A400 flex md:flex-col flex-row md:gap-5 items-start justify-end pb-1.5 px-1.5 w-full">
                <Img
                  className="button common-pointer absolute h-[45px] w-[45px] left-4 md:mt-0 mt-[26px]"
                  src="images/img_arrowdown.svg"
                  alt="arrowdown"
                  onClick={() => navigate('/frontpagelight')}
                />
                <Img className="absolute font-yogi font-lighter left-[8%] mt-[1%] w-[10%] tracking-[-0.30px]"
                  src="/images/teamlogo.png">
                </Img>
                <Text className="absolute font-yogi font-lighter left-[43%] mt-[1.7%] md:text-3xl text-4xl text-white-A700 tracking-[-0.30px]">
                  BASIC MODE
                </Text>
                <Img
                  src='/images/wpcjf.png'
                  className='button absolute w-[80px] mr-[10%] mt-[0.2%]'
                  onClick={redirectToSeasonMarket}></Img>
                <Text
                  className="ml-9 md:ml-[0] mr-[2%] md:mt-0 mt-[25px] md:text-3xl sm:text-[28px] text-[32px] text-center text-white-A700 tracking-[-0.16px]"
                  size="txtInterSemiBold32"
                >
                  <TimeComponent />
                </Text>
              </div>
              <div className="flex md:flex-col flex-row font-yogi md:gap-10 items-start justify-between mx-auto md:px-5 w-full">

                <Sidebar className="!sticky !w-[90px] bg-gray-50 flex !h-screen justify-start overflow-auto top-[0]">
                  <div className="flex flex-col items-center justify-start mt-[105%] mx-auto w-[48%]">
                    <div className="flex flex-col items-start justify-start w-[80%] align-center" >
                      <Img
                        className="search"
                        src="images/img_search.svg"
                        alt="rewind"
                        onClick={() => navigate("/familyhomelight/search")}
                      />
                      <Text
                        className="ml-1 md:ml-[0] text-[15px] text-center text-gray-800 sm:text-lg md:text-xl tracking-[-2px]"
                        size="txtYogi"
                        onClick={() => navigate("/familyhomelight/search")}
                      >
                        검색
                      </Text>

                    </div>

                    <div className="common-pointer flex flex-col items-center justify-start mt-[12px] w-full"
                      id='top-button'>
                      <Img
                        className="w-[55px] button"
                        src="images/top.png"
                        alt="to-top"
                        onClick={handleClick}
                      />
                      <Text
                        className="text-[15px] mt-[-1%] text-center font-yogi text-gray-800 sm:text-lg md:text-xl tracking-[-0.5px]"
                        onClick={handleClick}
                      >
                        TOP
                      </Text>
                    </div>
                    <div
                      className="common-pointer flex flex-col items-center justify-start mt-[12px] w-full"
                      id="home-button"

                    >
                      <Img
                        className="h-[55px] button"
                        src="images/img_home.svg"
                        alt="home"
                        onClick={scrollToRec}


                      />
                      <Text
                        className="text-[15px] mt-[-1%] text-center font-yogi text-gray-800 sm:text-lg md:text-xl tracking-[-2px]"
                        size="txtYogi"
                        onClick={scrollToRec}
                      >
                        추천 홈
                      </Text>
                    </div>

                    <div className="flex flex-col items-center justify-start mt-[12px] w-full"
                      id="drama-button">
                      <Img

                        className="h-[55px] button"
                        src="images/img_star.svg"
                        alt="star"
                        onClick={scrollToDrama}
                      />
                      <Text
                        className="text-[15px] mt-[-1%] text-center font-yogi text-gray-800 md:text-xl tracking-[-1px]"
                        size="txtYogi"
                        onClick={scrollToDrama}

                      >
                        드라마
                      </Text>
                    </div>
                    <div className="flex flex-col gap-0 items-center justify-start mt-[12px] w-[100%] md:w-full"
                      id="movie-button">
                      <Img
                        className="h-[55px] button"
                        src="images/img_thumbsup.svg"
                        alt="lock"
                        onClick={scrollToMovie}
                      />
                      <Text
                        className="text-[15px] mt-[-1%] text-center text-gray-800 sm:text-lg md:text-xl tracking-[-0.11px]"
                        size="txtYogi"
                        onClick={scrollToMovie}
                      >
                        영화
                      </Text>
                    </div>
                  </div>
                </Sidebar>
                <div>

                </div>
                {loading ? (
                  <div className='overlay'>
                    <LoadingScreen />
                  </div>
                ) : (

                  <div className="flex flex-1 flex-col items-start justify-start w-full mb-[10%]">
                    <div className="flex flex-1 flex-col items-start justify-start w-full">
                      <div className="flex flex-col items-center justify-start" style={{ marginTop: '1%' }}>
                        <Text
                          className="leading-[10.00px] ml-[5.5%] pl-[50px] sm:text-[21px] md:text-[23px] text-[25px] text-black-900 tracking-[-0.13px] w-full font-yogi"
                          size="txtYogi"
                        >
                          <span className="text-black-900 font-yogi text-left">
                            이달의 최신작{" "}
                          </span>
                          <span className="md:text-[46px] sm:text-[40px] text-red-A400 font-yellowtail text-left text-[50px] font-normal">
                            New{" "}
                          </span>
                        </Text>
                      </div>
                      <div className="flex md:flex-col flex-row font-paytoneone items-center justify-start w-full">
                        <div className="flex flex-col items-center justify-start" style={{ marginTop: '2%' }}>


                        </div>


                        <Firstlist></Firstlist>



                      </div>

                    </div>

                    <div className="flex flex-1 flex-col items-start justify-start w-full">
                      <div className="flex flex-col items-center justify-start" style={{ marginTop: '6%' }}>
                        <Text
                          className="leading-[10.00px] pl-[50px] ml-[5%] sm:text-[30px] md:text-[50px] text-[25px] text-black-900 tracking-[-0.13px] w-full font-yogi inline-block"
                          size="txtYogi"
                        >
                          <span className="text-black-900 font-yogi">
                            주간 베스트{" "}
                          </span>
                          <span className="md:text-[46px] sm:text-[40px] text-red-A400 font-yellowtail text-[40px] font-normal ">
                            Ranking{" "}
                          </span>
                        </Text>
                      </div>
                      <div className="flex md:flex-col flex-row font-paytoneone md:gap-5 items-start justify-between pr-[100px] w-full">
                        <div className="flex-shrink-0 h-[250px] relative w-1/6 mr-[10%] mb-[2%] md:w-full mt-[1%]">

                          <div className="video-container">
                            <Reclist rankposter={rankposter} />
                          </div>
                        </div>
                      </div>
                    </div>

                    {yetposter !== null && yetposter.length > 0 && (
                      <div className="flex flex-1 flex-col items-start justify-start w-full pb-[200px]">
                        <div className="flex flex-col items-center justify-start" style={{ marginTop: '6%' }}>
                          <Text
                            className="leading-[10.00px] pl-[50px] ml-[5%] sm:text-[30px] md:text-[50px] text-[25px] text-black-900 tracking-[-0.13px] w-full font-yogi inline-block"
                            size="txtYogi"
                          >

                            <span className="text-black-900 text-left font-yogi">
                              당신이 시청중인{" "}
                            </span>
                            <span className="md:text-[46px] sm:text-[40px] text-red-A400 font-yellowtail text-left text-[50px] font-normal">
                              VOD{" "}
                            </span>
                          </Text>
                        </div>
                        <div className="flex md:flex-col flex-row md:gap-5 items-start justify-between pr-[100px] w-full">


                          <div className="flex-shrink-0 h-[250px] relative w-1/6 md:w-full mb-[2%]">

                            <div className="video-container">
                              <Reclist rankposter={yetposter} />
                            </div>

                          </div>
                        </div>
                      </div>
                    )}

                    {/* 메인추천 */}
                    <div className="flex flex-col items-center justify-start "
                      style={{ marginTop: '10%' }} id='rec-page'>
                    </div>
                    <div className="bar flex flex-col items-center justify-start "  >
                      <Text
                        className="leading-[100.00px] pl-[50px] ml-[1%] sm:text-[21px] md:text-[23px] text-[30px] text-black-900 tracking-[-0.13px] w-full"
                        size="txtYogi"
                      >
                        <span className="text-black-900 text-left font-yogi">
                          추천 홈
                        </span>
                      </Text>
                    </div>
                    <div className="flex flex-1 flex-col items-start justify-start w-full">
                      <div className="flex flex-col items-center justify-start w-[100%]" style={{ marginTop: '2%', marginBottom: '-1.5%' }}>
                        <Text
                          className="leading-[100.00px] ml-[1.4%] pl-[50px] sm:text-[21px] md:text-[23px] text-[25px] text-black-900 tracking-[-0.13px] w-full"
                          size="txtYogi"
                        >
                          <span className="text-black-900 font-yogi text-left">
                            {localStorage.getItem('new') ? `당신이 선택한 장르 기반 추천 ` : `당신의 취향저격 장르 추천 `}
                          </span>
                          <span className="md:text-[46px] sm:text-[40px] text-red-A400 font-yellowtail text-left text-[50px] font-normal">
                            {localStorage.getItem('new') ? `Genre` : `Genre `}
                          </span>

                        </Text>
                        <div className='absolute rerec ml-[84%] mt-[1.5%]'>

                          <Img
                            className="h-[30px] but"
                            src="images/img_qrcode.svg"
                            alt="contrast"
                            onClick={Rerec}
                          />
                        </div>

                      </div>
                      <div className="flex md:flex-col flex-row font-paytoneone md:gap-5 items-start justify-between pr-[100px] w-full">
                        <div className="flex-shrink-0 h-[250px] relative w-1/6 mr-[10%] mb-[2%] md:w-full">

                          <div className="video-container">
                            <Reclist rankposter={genposter} />
                          </div>

                        </div>
                      </div>
                    </div>
                    <div className="flex flex-1 flex-col items-start justify-start w-full">
                      <div className="flex flex-col items-center justify-start w-[100%]" style={{ marginTop: '6%', marginBottom: '-1.5%' }}>
                        <Text
                          className="leading-[100.00px] ml-[1.4%] pl-[50px] sm:text-[21px] md:text-[23px] text-[25px] text-black-900 tracking-[-0.13px] w-full"
                          size="txtYogi"
                        >
                          <span className="text-black-900 text-left font-yogi">
                            오직 당신 만을 위한 추천!{" "}
                          </span>
                          <span className="md:text-[46px] sm:text-[40px] text-red-A400 font-yellowtail text-left text-[50px]">
                            For You{" "}
                          </span>


                        </Text>
                        <div className='absolute rerec ml-[84%] mt-[1.5%]'>
                          <Img
                            className="h-[30px] but"
                            src="images/img_qrcode.svg"
                            alt="contrast"
                            onClick={Rerec1}
                          />
                        </div>
                      </div>

                      <div className="flex md:flex-col flex-row md:gap-5 items-start justify-between pr-[100px] w-full">


                        <div className="flex-shrink-0 h-[250px] relative w-1/6 md:w-full mb-[2%]">

                          <div className="video-container">
                            <Reclist rankposter={userposter} />
                          </div>

                        </div>
                      </div>
                    </div>

                    <div className="flex flex-1 flex-col items-start justify-start w-full">
                      <div className="flex flex-col items-center justify-start w-[100%]" style={{ marginTop: '6%', marginBottom: '-1.5%' }}>
                        <Text
                          className="leading-[100.00px] ml-[1.4%] pl-[50px] sm:text-[21px] md:text-[23px] text-[25px] text-black-900 tracking-[-0.13px] w-full"
                          size="txtYogi"
                        >
                          <span className="text-black-900 text-left font-yogi">
                            이불속 드라마 정주행 추천!{" "}
                          </span>
                          <span className="md:text-[46px] sm:text-[40px] text-red-A400 font-yellowtail text-left text-[50px]">
                            Drama{" "}
                          </span>
                        </Text>

                        <div className='absolute rerec ml-[84%] mt-[1.5%]'>
                          <Img
                            className="h-[30px] but"
                            src="images/img_qrcode.svg"
                            alt="contrast"
                            onClick={Rerec3}
                          />
                        </div>
                      </div>
                      <div className="flex md:flex-col flex-row md:gap-5 items-start justify-between pr-[100px] w-full">


                        <div className="flex-shrink-0 h-[250px] relative w-1/6 md:w-full mb-[2%]">

                          <div className="video-container">
                            <Reclist rankposter={recposter1} />
                          </div>

                        </div>
                      </div>
                    </div>

                    {recposter !== null && recposter.length > 0 && (
                      <div className="flex flex-1 flex-col items-start justify-start w-full">
                        <div className="flex flex-col items-center justify-start w-[100%]" style={{ marginTop: '6%', marginBottom: '-1.5%' }}>
                          <Text
                            className="leading-[100.00px] ml-[1.4%] pl-[50px] sm:text-[21px] md:text-[23px] text-[25px] text-black-900 tracking-[-0.13px] w-full"
                            size="txtYogi"
                          >
                            <span className="text-black-900 font-yogi text-left font-normal">

                              최근 시청한&nbsp;
                            </span>
                            <span className="text-red-A400 font-yogi text-left font-normal text-[30px]">

                              '{current}'&nbsp;
                            </span>
                            <span className="text-black-900 font-yogi text-left font-normal">

                              연관 콘텐츠 추천! &nbsp;
                            </span>
                            <span className="md:text-[46px] sm:text-[40px] text-red-A400 font-yellowtail text-left text-[50px] font-normal">
                              VOD{" "}
                            </span>
                          </Text>
                          <div className='absolute rerec ml-[84%] mt-[1.5%]'>
                            <Img
                              className="h-[30px] w-[30px] but"
                              src="images/img_qrcode.svg"
                              alt="contrast"
                              onClick={Rerec2}
                            />
                          </div>
                        </div>
                        <div className="flex md:flex-col flex-row items-start justify-between pr-[100px] w-full">


                          <div className="flex-shrink-0 h-[250px] relative w-1/6 md:w-full mb-[2%]">

                            <div className="video-container">
                              <Reclist rankposter={recposter} />
                            </div>

                          </div>
                        </div>
                      </div>
                    )}



                    {/* 드라마 구간 */}
                    <div className="flex flex-col items-center justify-start "
                      style={{ marginTop: '10%' }} id='drama-page'>
                    </div>
                    <div className="bar flex flex-col items-center justify-start " >
                      <Text
                        className="leading-[100.00px] pl-[50px] ml-[1%] sm:text-[21px] md:text-[23px] text-[30px] text-black-900 tracking-[-0.13px] w-full"
                        size="txtYogi"
                      // id="drama-page"
                      >
                        <span className="text-black-900 text-left font-yogi">
                          드라마{" "}
                        </span>

                      </Text>
                    </div>

                    <div className="flex flex-1 flex-col items-start justify-start w-full">
                      <div className="flex flex-col items-center justify-start " style={{ marginTop: '2%' }}>
                        <Text
                          className="leading-[100.00px] ml-[3%] pl-[50px] mb-[-4%] sm:text-[21px] md:text-[23px] text-[25px] text-black-900 tracking-[-0.13px] w-full"
                          size="txtYogi"
                        >
                          <span className="text-black-900 text-left font-yogi">
                            이주의 드라마{" "}
                          </span>
                          <span className="md:text-[46px] sm:text-[40px] text-red-A400 font-yellowtail text-left text-[50px]">
                            Ranking{" "}
                          </span>
                        </Text>
                      </div>
                      <div className="flex md:flex-col flex-row md:gap-5 items-start justify-between pr-[100px] w-full">


                        <div className="flex-shrink-0 h-[250px] relative w-1/6 md:w-full mb-[2%]">

                          <div className="video-container">
                            <Reclist rankposter={rankposter1} />
                          </div>

                        </div>
                      </div>
                    </div>





                    <Drama ctcl={ctcl1} ctclname={ctclname} i={0} />
                    <Drama ctcl={ctcl2} ctclname={ctclname} i={1} />
                    <Drama ctcl={ctcl3} ctclname={ctclname} i={2} />
                    <Drama ctcl={ctcl4} ctclname={ctclname} i={3} />
                    <Drama ctcl={ctcl5} ctclname={ctclname} i={4} />
                    <Drama ctcl={ctcl6} ctclname={ctclname} i={5} />




                    {/*영화 */}
                    <div className="flex flex-col items-center justify-start "
                      style={{ marginTop: '10%' }} id='movie-page'>
                    </div>
                    <div className="bar flex flex-col items-center justify-start">
                      <Text
                        className="leading-[100.00px] pl-[50px] ml-[1%] sm:text-[21px] md:text-[23px] text-[30px] text-black-900 tracking-[-0.13px] w-full"
                        size="txtYogi"
                      >
                        <span className="text-black-900 text-left font-yogi">
                          영화{" "}
                        </span>

                      </Text>
                    </div>
                    <div className="flex flex-1 flex-col items-start justify-start w-full">
                      <div className="flex flex-col items-center justify-start " style={{ marginTop: '2%' }}>
                        <Text
                          className="leading-[100.00px] ml-[3%] pl-[50px] mb-[-4%] sm:text-[21px] md:text-[23px] text-[25px] text-black-900 tracking-[-0.13px] w-full"
                          size="txtYogi"
                        >
                          <span className="text-black-900 text-left font-yogi">
                            주간 영화 베스트{" "}
                          </span>
                          <span className="md:text-[46px] sm:text-[40px] text-red-A400 font-yellowtail text-left text-[50px]">
                            Best{" "}
                          </span>
                        </Text>
                      </div>
                      <div className="flex md:flex-col flex-row md:gap-5 items-start justify-between pr-[100px] w-full">


                        <div className="flex-shrink-0 h-[250px] relative w-1/6 md:w-full mb-[2%]">

                          <div className="video-container">
                            <Reclist rankposter={rankposter11} />
                          </div>

                        </div>
                      </div>
                    </div>

                    <Mov ctcl={ctcl11} ctclname={ctclname1} i={0} />
                    <Mov ctcl={ctcl21} ctclname={ctclname1} i={1} />
                    <Mov ctcl={ctcl31} ctclname={ctclname1} i={2} />
                    <Mov ctcl={ctcl41} ctclname={ctclname1} i={3} />
                    <Mov ctcl={ctcl51} ctclname={ctclname1} i={4} />
                    <Mov ctcl={ctcl61} ctclname={ctclname1} i={5} />
                    <Mov ctcl={ctcl71} ctclname={ctclname1} i={6} />







                    <div className="flex flex-col items-center justify-start "
                      style={{ marginTop: '10%' }} id='end-page'>
                    </div>
                    <button className='top1' style={{ width: '70px', height: '70px' }}
                      onClick={handleClick1} id='totop'>TOP</button>


                  </div>



                )}

              </div>
            </div>

            <Text
              className="leading-[100.00px] pl-[50px]  sm:text-[21px] md:text-[23px] text-[10px] text-black-900 tracking-[-0.13px] w-full"
              style={{ backgroundColor: 'gray', width: '100%' }}
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
        </div>
      </CSSTransition>
    </>
  );
};

export default FamilyHomeLightPage;





