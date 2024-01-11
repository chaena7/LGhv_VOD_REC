import React, { useState, useEffect } from "react";
import { CSSTransition } from 'react-transition-group';
import { useNavigate } from "react-router-dom";
import 'styles/animation.css'
import { Img, Text, TimeComponent } from "components";
import 'styles/img.css'



const FrontpageLightPage = () => {
  const [showPage, setShowPage] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    setShowPage(true);
  }, []);

  const handleLogout = () => {
    if (localStorage.getItem('new') === null) {
      const confirmLogout = window.confirm('로그아웃 하시겠습니까?');
      if (confirmLogout) {


        localStorage.clear();
        document.cookie.split(";").forEach(cookie => {
          document.cookie = cookie.replace(/^ +/, "").replace(/=.*/, `=;expires=${new Date().toUTCString()};path=/`);

        });

        navigate('/');
      } else {

      }

    } else {

      navigate('/SignupPageLight');
    }
  }


  const access = localStorage.getItem('access_token');
  if (!access) {
    navigate('/');
    return; // 로그인 페이지로 이동 후 함수 종료
  }

  return (
    <>
      <CSSTransition
        in={showPage}
        timeout={3000}
        classNames="fade"
        unmountOnExit
      >
        <div className="bg-gray-100 border border-black-900 border-solid flex flex-col font-inter items-center justify-start mx-auto w-full">
          <div className="flex flex-col items-center justify-start mb-[20px] w-full">
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
                className="ml-9 md:ml-[0] mr-[2%] md:mt-0 mt-[25px] md:text-3xl sm:text-[28px] text-[32px] text-center text-white-A700 tracking-[-0.16px]"
                size="txtInterSemiBold32"
              >
                <TimeComponent />
              </Text>
            </div>

            <Text
              className="font-yogi mt-[70px] text-3xl sm:text-[26px] md:text-[28px] text-black-900 text-center tracking-[-0.15px]"
              size="txtYogi0"
            >
              시청할 모드를 선택해주세요.
            </Text>
            <div className="flex md:flex-col flex-row md:gap-10 gap-[90px] items-center justify-start max-w-[1100px] mt-[30px] mx-auto md:px-5 w-full">
              <div className="button2 md:h-[640px] h-[640px] relative w-[50%] md:w-full">
                <div
                  className="common-pointer absolute bg-white-A700 flex flex-col h-full inset-y-[0] items-center justify-center left-[0] my-auto outline outline-[1px] outline-white-A700_4c p-[27px] sm:px-5 rounded-[40px] shadow-bs h-[95%]"
                  onClick={() => navigate("/familyhomelight")}
                >
                  <Text
                    className="mt-[97px] md:text-5xl text-[100px] text-red-A400 tracking-[0.16px]"
                    size="txtYellowtailRegular100"
                  >
                    Basic
                  </Text>
                  <Text
                    className="mb-[322px] sm:text-[21px] md:text-[23px] text-[25px] text-black-900 tracking-[-0.13px] font-yogi"
                    size="txtYogi"
                  >
                    많은 기능을 다양하게 이용하고 싶다면?
                  </Text>
                </div>
                <div className="absolute bottom-[6%] h-[239px] right-[0] w-[93%] sm:w-full">
                  <div className="absolute h-[239px] inset-[0] justify-center m-auto w-[97%] sm:w-full">
                    <Img
                      className="h-[163px] ml-auto mr-[81px] mt-[9px] object-cover rounded-[15px] w-[26%]"
                      src="images/img_.png"
                      alt="thirteen"
                    />
                    <div className="absolute h-[239px] inset-[0] justify-center m-auto w-full">
                      <Img
                        className="h-[239px] m-auto object-cover w-full"
                        src="images/img__239x481.png"
                        alt="fourteen"
                      />
                      <Img
                        className="absolute h-[162px] left-[15%] object-cover rounded-[15px] top-[4%] w-[26%]"
                        src="images/img__162x122.png"
                        alt="fifteen"
                      />
                    </div>
                  </div>
                  <Img
                    className="absolute h-[163px] left-[0] object-cover rounded-[15px] top-[13%] w-1/4"
                    src="images/img__163x122.png"
                    alt="sixteen"
                  />
                </div>
              </div>
              <div
                className="button2 common-pointer bg-white-A700 flex md:flex-1 flex-col items-center justify-end outline outline-[1px] outline-white-A700_4c pt-3 md:px-5 px-12 sm:px-5 rounded-[40px] shadow-bs w-[50%] h-[50%] md:w-full"
                onClick={() => navigate("/simple")}
              >
                <Text
                  className="mt-[70px] md:text-5xl text-[100px] text-red-A400 tracking-[0.16px]"
                  size="txtYellowtailRegular100"
                >
                  Simple
                </Text>
                <Text
                  className="font-yogi sm:text-[21px] md:text-[23px] text-[24px] text-black-900 tracking-[-0.5px]"
                  size="txtYogi"
                >
                  핵심 기능만 크게 이용하고 싶다면?
                </Text>
                <Img
                  className="h-[318px] sm:h-auto mt-10 object-cover w-[85%] md:w-full"
                  src="images/img__318x320.png"
                  alt="eighteen"
                />
              </div>
            </div>
          </div>
          <Text
            className="leading-[100.00px] pl-[50px]  sm:text-[21px] md:text-[23px] text-[10px] text-black-900 tracking-[-0.13px] w-full"
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

export default FrontpageLightPage;
