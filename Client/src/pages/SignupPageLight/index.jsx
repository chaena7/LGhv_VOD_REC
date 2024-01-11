import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button, Img, List, Text, TimeComponent } from "components";
import '../../styles/button.css'
import { CSSTransition } from "react-transition-group";

const SignupPageLightPage = () => {
  const handleLogout = () => {
    const confirmLogout = window.confirm('로그아웃 하시겠습니까?');
    if (confirmLogout) {
      localStorage.clear();
      document.cookie.split(";").forEach(cookie => {
        document.cookie = cookie.replace(/^ +/, "").replace(/=.*/, `=;expires=${new Date().toUTCString()};path=/`);
      });

      navigate('/');
    } else {

    }
  };
  const [showPage, setShowPage] = useState(false);
  const [id, setId] = useState('');
  const [selectedGenres, setSelectedGenres] = useState([]);
  const genres = ['드라마', '액션', '모험', '코미디', '로맨스', '애니메이션', '스릴러', '해외시리즈', '멜로', '판타지',
    '공포', '범죄', 'SF', '미스터리', '가족', '예능', '시대극', '다큐', '시사교양', '키즈']
  // const [genre, setGenre] = useState([]);
  // const [buttonClassName, setButton] = useState(); 
  useEffect(() => {
    setShowPage(true)
  }, [])

  useEffect(() => {
    const storedId = localStorage.getItem('subsr');
    if (storedId) {
      setId(storedId);
    }
  }, []);
  const navigate = useNavigate();
  // useEffect(() => {
  //   const isSelected = selectedGenres.includes(genre);
  //   const button =isSelected ? 'selected-class' : 'default-class';
  //   setButton(button);
  // }, [selectedGenres, genre]);

  const handleGenreClick = (clickedGenre) => {
    let updatedGenres = [...selectedGenres];

    if (selectedGenres.includes(clickedGenre)) {
      updatedGenres = selectedGenres.filter((genre) => genre !== clickedGenre);
    } else {
      updatedGenres = [...selectedGenres, clickedGenre];
    }

    setSelectedGenres(updatedGenres);
    localStorage.setItem('genre', JSON.stringify(updatedGenres.map(genre => genres.indexOf(genre))));
  };


  const isSelected = (genre) => selectedGenres.includes(genre);
  const getButton = (genre) => isSelected(genre) ? 'selected-class' : 'default-class';




  return (
    <>
      <CSSTransition
        in={showPage}
        timeout={4000}
        classNames="fade"
        unmountOnExit
      >
        <div className="bg-gray-100 border border-black-900 border-solid flex flex-col font-inter sm:gap-10 md:gap-10 gap-[138px] justify-start mx-auto w-full">
          <div className="flex flex-col items-center w-full ">
            <div className="z-10 !sticky top-[0] overflow-block relative flex bg-red-A400 flex md:flex-col h-[90px] flex-row md:gap-5 items-start justify-end pb-1.5 px-1.5 w-full">
              <Img
                className="button common-pointer absolute h-[45px] w-[45px] left-5 md:mt-0 mt-[26px]"
                src="images/img_arrowdown.svg"
                alt="arrowdown"
                onClick={() => handleLogout(true)}
              />

              <Img className="absolute font-yogi font-lighter left-[8%] mt-[1%] w-[10%] tracking-[-0.30px]"
                src="/images/teamlogo.png">
              </Img>

              <Text
                className="ml-9 md:ml-[0] mr-[15px] md:mt-0 mt-[25px] md:text-3xl sm:text-[28px] text-[32px] text-center text-white-A700 tracking-[-0.16px]"
                size="txtInterSemiBold32"
              >
                <TimeComponent />
              </Text>
            </div>
            <Text
              className="leading-[50.00px] mt-[50px] text-3xl sm:text-[32px] md:text-[34px] text-black-900 text-center tracking-[-0.18px]"
              size="txtInterSemiBold36"
            >
              <span className="text-black-900 font-inter font-yogi">
                안녕하세요,{" "}
              </span>
              <span className="text-red-A400 text-[40px] font-inter font-yogi">
                {id}
              </span>
              <span className="text-black-900 font-inter font-yogi">
                <>
                  {" "}
                  님<br />
                  선호하는 장르를 모두 골라주세요.{" "}
                </>
              </span>
            </Text>
            <div className="flex flex-col items-center justify-start max-w-[921px] mt-[3%] mx-auto md:px-5 w-full">
              <div className="flex flex-col md:gap-10 gap-16 items-start justify-start w-full">
                <List
                  className="flex flex-col gap-13 items-center w-full"
                  orientation="horizontal"
                >
                  {/* <div className="flex flex-1 md:flex-col flex-row md:gap-10 items-center justify-between my-0 w-full"> */}
                  {/* <button
                    className={getButton(genre)}
                    // shape="round"
                    // color="deep_orange_50_fc"
                    // variant="fill"
                    onClick={() => handleGenreClick("액션")}
                  >
                    액션
                  </button> */}
                  <div className="button-container font-yogi">
                    {genres.map((genre) => (
                      <button
                        key={genre}
                        className={getButton(genre)}
                        onClick={() => {
                          handleGenreClick(genre);
                          localStorage.setItem('new', 'new')
                        }
                        }
                      >
                        {genre}
                      </button>
                    ))}
                  </div>
                </List>


              </div>
            </div>
          </div>
          <div className="flex flex-col items-center mt-[-5%] w-full">
            <Button
              className="common-pointer cursor-pointer font-yogi h-[50px] sm:text-2xl md:text-[26px] text-[28px] text-center tracking-[-0.14px] w-[158px]"
              onClick={() => {
                const storedGenres = JSON.parse(localStorage.getItem('genre') || '[]');
                if (storedGenres.length === 0) {
                  alert('장르를 1개 이상 골라주세요.');
                } else {
                  navigate("/FrontpageLight");
                }
              }}
              shape="round"
              color="red_A400"
              variant="fill"
            >
              완료
            </Button>
          </div>
          <Text
            className="leading-[100.00px] pl-[50px]  sm:text-[21px] md:text-[23px] text-[10px] text-black-900 tracking-[-0.13px] mt-[3%] w-full"
            style={{ backgroundColor: 'gray', width: '100%' }}
          >
            <span className="text-black-900 font-yogi mt-[5%]" style={{ display: 'block', marginBottom: '8px' }}>
              Made by HELLO00 Front 황성주
            </span>
            <span className="text-black-900 font-yogi mt-[1%]" style={{ display: 'block', marginBottom: '8px' }}>
              Member: 공유경, 김명현, 김은혜, 박채나, 황성주
            </span>
            <span className="text-black-900 font-yogi mt-[2%] mb-[5%]" style={{ display: 'block' }}>
              Studied at LG Hello Vision DX DATA SCHOOL
            </span>
          </Text>
        </div>
      </CSSTransition>
    </>
  );
};

export default SignupPageLightPage;
