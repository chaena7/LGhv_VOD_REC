import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from 'axios';
import { Button, Img, Text, TimeComponent } from "components"; //Input, Text
import C1LoginPageLightIdinput from "components/C1LoginPageLightIdinput/login_input";
import { CSSTransition } from "react-transition-group";
// import { useCookies } from 'react-cookie';
import 'styles/animation.css'
import 'styles/img.css'
const LoginPageLightPage = () => {
  const navigate = useNavigate();
  const [accessToken, setAccessToken] = useState('');
  const [showPage, setShowPage] = useState(false);
  const [id, setId] = useState('');

  useEffect(( )=> {
    setShowPage(true);
  }, []);

  useEffect(() => {
    const storedToken = localStorage.getItem('access_token');

    if (storedToken) {
      // access_token이 있다면 원하는 URL로 이동
      navigate('/FrontpageLight');
      setAccessToken(storedToken)
    }
  }, [navigate]);

  const handleInputChange = (e) => {
    //alert(e)
    //const newValue = e.target.value;
    setId(e);
    console.log(e);
  }
  // const [cookie, setCookie] = useCookies(['access']);


const handleLogin = async () => {
  try {
    
    const response = await axios.get('https://hello00back.net/vodrec/', {
      headers: {
        Authorization : accessToken,
      }
    });
    console.log(accessToken);
    // const newCookie = response.data.token;
    // setCookie('access', newCookie);
      // console.log(response.status);
      if (response.status === 200){
        navigate('FrontpageLight');
        // console.log(cookie);
      }
    
    }catch (error) {
    console.error('Error 발생 :', error);
    
    if (error.response.status !== 200){
      
      try {
        const postresponse = await axios.post('https://hello00back.net/login/', {
          'subsr': id
        });
        
        if (postresponse.status === 200 && postresponse.data.message === 0) {
          
          console.log(postresponse.data)
          // setCookie('access', postresponse.data.token.access)
          const accessToken = postresponse.data.token.access; // 예상되는 응답에서 access_token을 가져옴
          const refreshToken = postresponse.data.token.refresh;
          // 쿠키 및 로컬 스토리지에 토큰 저장
          localStorage.setItem('ip', postresponse.data.ip);
          localStorage.setItem('access_token', accessToken);
          localStorage.setItem('refresh_token', refreshToken);
          localStorage.setItem('subsr', postresponse.data.user.subsr);
          localStorage.setItem('subsr_id', postresponse.data.user.subsr_id);
          // axios의 인스턴스에 헤더 추가
          axios.defaults.headers.common['Authorization'] = 'JWT ' + accessToken;
          console.log("로그인 성공적")
 
          // navigate(""); // 페이지 이동
          // location.reload(); // 페이지 새로고침
          navigate('FrontpageLight'); // 원하는 페이지로 이동
          
        } else if (postresponse.status === 200 && postresponse.data.message === 1) {
          const accessToken1 = postresponse.data.token.access; // 예상되는 응답에서 access_token을 가져옴
          const refreshToken1 = postresponse.data.token.refresh;
          console.log(postresponse.data);
          // setCookie('access', postresponse.data.token.access)

          // // 쿠키 및 로컬 스토리지에 토큰 저장
          localStorage.setItem('ip', postresponse.data.ip);
          localStorage.setItem('access_token', accessToken1);
          localStorage.setItem('refresh_token', refreshToken1);
          localStorage.setItem('subsr', id);
          localStorage.setItem('subsr_id', postresponse.data.user.subsr_id);
          // // axios의 인스턴스에 헤더 추가
          // axios.defaults.headers.common['Authorization'] = 'JWT ' + accessToken;

          
          navigate('/signuppagelight');
          
        };
      
      } catch (error) {
        console.error('로그인 POST 요청 실패:', error.response.data);
        setId('');
        const error_mess = error.response.data.message;
        alert(error_mess);
      }
    } else {
      console.log(error.data);
      console.error('GET 요청 실패:', error.response);
      setId('');
      const error_mess = error.response.data.message;
      alert(`${error_mess}`);
    }
  }
};
  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      // e.preventDefault();
      // e.stopPropagation();
      handleLogin();
    } 
  };

  return (
    <>
      <CSSTransition
        in={showPage}
        timeout={2000}
        classNames="fade"
       unmountOnExit
      >
      <div className="bg-gray-100 border border-black-800 border-solid flex flex-col sm:gap-10 md:gap-10 gap-[156px] items-center justify-start mx-auto w-full">
        <div className="z-10 !sticky top-[0] overflow-block h-[90px] relative flex bg-red-A400 flex md:flex-col flex-row md:gap-5 items-start justify-end pb-1.5 px-1.5 w-full">

            <Img className="absolute font-yogi font-lighter left-[2%] mt-[1%] w-[10%] tracking-[-0.30px]"
            src="/images/teamlogo.png">
            </Img>


            <Text
              className="ml-9 md:ml-[0] mr-[2%] md:mt-0 mt-[25px] md:text-3xl sm:text-[28px] text-[32px] text-center text-white-A700 tracking-[-0.16px]"
              size="txtInterSemiBold32"
            >
              <TimeComponent/>
            </Text>
          </div>
        <div className="md:h-[458px] h-[573px] md:px-5 relative w-[55%] md:w-full">
          <div className="absolute bg-white-A700 bottom-[10] flex flex-col inset-x-[1] items-center justify-center mx-auto outline outline-[1px] outline-white-A700_4c p-[41px] md:px-10 sm:px-5 rounded-[40px] shadow-bs w-full">
            <Img
              className="absolute h-[200px] inset-x-[0] mx-auto object-cover top-[-150px] w-[30%]"
              src="images/img_hello.png"
              alt="hello"
              style={{ objectFit: 'contain' }}
            />

            <C1LoginPageLightIdinput
              className="flex flex-col font-ibmplexsans h-[124px] md:h-auto items-start justify-start max-w-[700px] mt-[100px] w-full"
              value={id}
              onChange = {handleInputChange}
              onKeyPress = {handleKeyPress}
              
            />
            <div className="flex flex-col font-inter items-start justify-start mt-0.5 w-auto"
              
              tabIndex={0}
            >
              <Button
                className="button2 common-pointer cursor-pointer font-yogi h-[50px] text-center text-lg tracking-[-0.09px] w-[175px]"
                shape="round"
                color="blue_gray_100"
                size="lg"
                variant="fill"
                type="submit"
                onClick={handleLogin}
              >
                로그인
              </Button>
            </div>
            <Button 
              className="bg-transparent cursor-pointer font-yogi mb-[62px] min-w-[113px] mt-[17px] text-[15px] text-black-900 text-center tracking-[0.16px]"
              
              // onClick={() => navigate('/signuppagelight/')} 
              >
              세탑 등록 후 이용해주세요
              
            </Button>
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

export default LoginPageLightPage;
