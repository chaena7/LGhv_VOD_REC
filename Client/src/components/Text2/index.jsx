import React from "react";

const sizeClasses = {
  txtInterSemiBold30Black900: "font-inter font-semibold",
  txtInterSemiBold24Bluegray900: "font-inter font-semibold",
  txtABeeZeeRegular22RedA400: "font-abeezee font-normal",
  txtInterSemiBold36: "font-inter font-semibold",
  txtPaytoneOneRegular100: "font-normal font-paytoneone",
  txtInterSemiBold24: "font-inter font-semibold",
  txtInterSemiBold35: "font-inter font-semibold",
  txtInterSemiBold28: "font-inter font-semibold",
  txtInterBold50: "font-bold font-inter",
  txtInterSemiBold32: "font-inter font-semibold",
  txtInterSemiBold30: "font-inter font-semibold",
  txtInterSemiBold20: "font-inter font-semibold",
  txtABeeZeeRegular25: "font-abeezee font-normal",
  txtYellowtailRegular100: "font-normal font-yellowtail",
  txtInterRegular15: "font-inter font-normal",
  txtABeeZeeRegular22: "font-abeezee font-normal",
  txtPaytoneOneRegular80: "font-normal font-paytoneone",
  txtYesevaOneRegular35: "font-normal font-yesevaone",
  txtYogi: "font-yogi font-noraml"
};

const Text2 = ({ children, className = "", size, as, ...restProps }) => {
  const Component = as || "p";

  return (
    <Component
      className={`text-left ${className} ${size && sizeClasses[size]}`}
      style = {{ color : 'white'}}
      {...restProps}
    >
      {children}
    </Component>
  );
};

export { Text2 };