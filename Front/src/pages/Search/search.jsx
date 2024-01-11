import React, { useState, useEffect } from 'react';
import useDebounce from './useDebounce';
import { useNavigate } from 'react-router-dom';
import { Img, Text, Weather, TimeComponent, Searchlist, Default, Voice } from "components";
import { CSSTransition } from "react-transition-group";
//import ReactModal from 'react-modal';
import axios from 'axios';
import 'styles/input.css';
import 'styles/animation.css';
import 'styles/voice.css';



function Searchpage() {
  const [search, setSearch] = useState('');
  const [vods, setVods] = useState([]);
  const [showPage, setShowPage] = useState(false);
  const debounceValue = useDebounce(search, 300);
  const navigate = useNavigate();

  const rank1 = ((JSON.parse(localStorage.getItem('rank1'))).slice(0, 7)).map(item => item);
  const rank2 = ((JSON.parse(localStorage.getItem('rank2'))).slice(0, 7)).map(item => item);
  const rank3 = ((JSON.parse(localStorage.getItem('rank3'))).slice(0, 7)).map(item => item);


  localStorage.setItem('page', 5);


  useEffect(() => {
    setShowPage(true);
  }, []);


  const name = localStorage.getItem('name');
  // const access = localStorage.getItem('access_token');


  const handleInputChange = (e) => {
    const inputValue = e.target.value;
    setSearch(inputValue);
    console.log(e.target.value);
    if (search === '' && debounceValue === '') {
      // 빈 값인데도 debounceValue가 빈 값이면 debounceValue에 빈 값이 아닌 이전 값으로 설정
      setSearch(previousSearch => previousSearch);
    }
  };

  const handleTranscript = (transcript) => {
    setSearch(transcript); // transcript 값을 search에 할당
  };



  let route = '/FamilyHomeLight'; // 기본 경로 설정

  if (name === '1') {
    route = '/simple';
  } else {
    route = '/familyhomelight';
  }

  // 클릭 이벤트 핸들러
  const handleClick = () => {
    navigate(route); // 설정된 경로로 이동
  };

  useEffect(() => {
    const fetchData = async () => {

      try {
        console.log(debounceValue);
        const response = await axios.get(`https://hello00back.net/search/?Searchword=${debounceValue}`);

        if (response.status === 200) {
          const videos = response.data.data;
          // console.log(videos);
          console.log(debounceValue);

          let limitedVods = (videos.slice(0, 21)).map(item => item); // Limit the videos to 20 elements
          setVods(limitedVods); // Set the state with the limited videos array

        } else {
          console.log("response 오류")
        }
      } catch (error) {
        console.error('Error:', error);
      }
    };
    fetchData();
  }, [debounceValue, search]);


  return (
    <>

      <div className="bg-gray-100 border border-black-900 border-solid flex flex-col font-yogi justify-start mx-auto w-full h-100vh" style={{
        backgroundSize: 'cover', // 배경을 요소에 맞게 확장
        backgroundRepeat: 'no-repeat', // 배경 반복 없음
        backgroundAttachment: 'fixed', // 배경 고정
      }}>
        <div className="flex flex-col justify-start w-full">
          <div className="z-10 !sticky top-[0] overflow-block relative flex bg-red-A400 flex md:flex-col flex-row md:gap-5 items-start justify-end pb-1.5 px-1.5 w-full">
            <Img
              className="button common-pointer absolute h-[45px] w-[45px] left-5 md:mt-0 mt-[26px]"
              src="/images/img_arrowdown.svg"
              alt="arrowdown"
              onClick={handleClick}
            />
            <Img className="absolute font-yogi font-lighter left-[8%] mt-[1%] w-[10%] tracking-[-0.30px]"
              src="/images/teamlogo.png">
            </Img>
            <Text className="absolute font-yogi font-lighter left-[43%] mt-[1.7%] md:text-3xl text-4xl text-white-A700 tracking-[-0.30px]">
              SEARCH
            </Text>
            <div className="relative h-[80px] w-[13%] mr-[1%] mt-[4px] w-[15%] md:w-full">
              <Weather />
            </div>
            <Text
              className="ml-9 md:ml-[0] mr-[2%] md:mt-0 mt-[25px] md:text-3xl sm:text-[28px] text-[32px] text-center text-white-A700 tracking-[-0.16px]"
              size="txtInterSemiBold32"
            >
              <TimeComponent />
            </Text>
          </div>
          <CSSTransition
            in={showPage}
            timeout={5000}
            classNames="fade"
            unmountOnExit
          >

            <div>
              <input
                type="text"
                value={search}
                onChange={handleInputChange}
                style={{ width: '100%', minHeight: '120px', display: 'flex', backgroundColor: 'rgb(175, 178, 190, 0.5)', paddingLeft: '5.5%' }}
                className='absolute !sticky input-search text-[45px] pl-10'
                placeholder="제목, 캐릭터 또는 장르로 검색하세요"
              />

              <div className='absolute mt-[-6%] ml-[90%]'>

                <Voice onTranscript={handleTranscript} />

              </div>

              <div className='ml-[5.5%] mt-[2%] mb-[10%] relative left-0 flex flex-col'>
                {search === '' ? (
                  <div>
                    <div className='font-yogi text-[23px]'>
                      주간 베스트
                      <Default poster={rank1} ></Default>
                    </div>
                    <div className='font-yogi text-[23px]'>
                      드라마 베스트
                      <Default poster={rank2} ></Default>
                    </div>
                    <div className='font-yogi text-[23px]'>
                      영화 베스트
                      <Default poster={rank3} ></Default>
                    </div>
                  </div>
                ) : vods.length > 0 ? (
                  <div className='pb-[50%] '>
                    <text className='font-yogi text-[25px]'>
                      "<span style={{ color: 'red' }}>{search}</span>" 에 대한 검색 결과
                    </text>
                    <Searchlist rankposter={vods} />
                  </div>
                ) : (
                  <div className="flex justify-center items-start h-screen">
                    <div className='font-yogi mt-[10%] text-[50px]'>
                      입력한 "<span style={{ color: 'red' }}>{search}</span>"에 대한 결과가 없습니다.
                    </div>
                  </div>

                )}

              </div>
            </div>
          </CSSTransition>
        </div>


      </div>

    </>
  );

};

export default Searchpage;