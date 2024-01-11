import React from 'react';
import "styles/smooth.css"
import "styles/list.css"
import { Text, Reclist, } from "components"

const Drama = ({ ctcl, ctclname, i }) => {

  
    return (
        <div className="flex flex-1 flex-col items-start justify-start w-full">
        <div className="flex flex-col items-center justify-start" style={{ marginTop: '6%' }}>
        <Text
            className="leading-[100.00px] ml-[3%] pl-[50px] mb-[-4%] sm:text-[21px] md:text-[23px] text-[25px] text-black-900 tracking-[-0.13px] w-full"
            size="txtYogi"
        >
            {ctclname[i] === '드라마' && (
            <span className="text-black-900 font-yogi text-left">
              TV/{ctclname[i]} 장르{" "}
            </span>
            )}
            {ctclname[i] !== '드라마' && (
            <span className="text-black-900 font-yogi text-left">
            {ctclname[i]} 드라마{" "}
            </span>
            )}
        <span className="md:text-[46px] sm:text-[40px] text-red-A400 font-yellowtail text-left text-[50px] font-normal">
        Genre{" "}
        </span>
        </Text>

        </div>
        <div className="flex md:flex-col flex-row md:gap-5 items-start justify-between pr-[100px] w-full">
                  
                 
            <div className="flex-shrink-0 h-[250px] relative w-1/6 md:w-full mb-[2%]">
                  
                <div className="video-container">
                <Reclist rankposter={ctcl} />
                </div>
                    
            </div>
            </div>
        </div>

        )
    };

export { Drama } ;