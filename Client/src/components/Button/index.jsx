import React from "react";
import PropTypes from "prop-types";

const shapes = { round: "rounded-[25px]" };
const variants = {
  fill: {
    blue_gray_100: "bg-blue_gray-100 text-black-900",
    white_A700_dd: "bg-white-A700_dd shadow-bs text-black-900",
    red_A400: "bg-red-A400 text-white-A700",
    deep_orange_50_fc: "bg-deep_orange-50_fc text-black-900",
  },
};
const sizes = { xs: "p-px", sm: "p-[5px]", md: "p-2", lg: "p-3.5" };

const Button = ({
  children,
  className = "",
  leftIcon,
  rightIcon,
  shape = "",
  size = "md",
  variant = "",
  color = "",
  
  ...restProps
}) => {

  return (
    <button
      className={`${className} ${(shape && shapes[shape]) || ""} ${
        (size && sizes[size]) || ""
      } ${(variant && variants[variant]?.[color]) || ""}`}
      {...restProps}
    >
      {!!leftIcon && leftIcon}
      {children}
      {!!rightIcon && rightIcon}
    </button>
  );
};

Button.propTypes = {
  className: PropTypes.string,
  children: PropTypes.node,
  shape: PropTypes.oneOf(["round"]),
  size: PropTypes.oneOf(["xs", "sm", "md", "lg"]),
  variant: PropTypes.oneOf(["fill"]),
  color: PropTypes.oneOf([
    "blue_gray_100",
    "white_A700_dd",
    "red_A400",
    "deep_orange_50_fc",
  ]),
  onKeyDown: PropTypes.func,
  onKeyPress: PropTypes.func,
  onKeyUp: PropTypes.func,
  onClick: PropTypes.func,
};

export { Button };
