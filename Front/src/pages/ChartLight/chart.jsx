import { ResponsivePie } from '@nivo/pie'
import { ResponsiveBar } from '@nivo/bar';
import { ResponsiveLine } from '@nivo/line'
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Text } from "components";




const CP = () => {
    const [gen_puser, setGen] = useState([]);
    const [m_performance, setM_Performance] = useState([]);
    const [ctcl_puser, setCtcl] = useState([]);
    const [conversion_rate, setCr] = useState([]);
    const [t_viewer, setTV] = useState();
    const [t_activeuser, setTAU] = useState();
    const [t_conversionrate, setTCR] = useState();
    const [t_maxprogram, setTMP] = useState();

    useEffect(() => {
        const getdata = async () => {
            try {
                const response = await axios.get('https://hello00back.net/chart_sample/');
                // const response = await axios.get('http://127.0.0.1:8000/chart_sample/');
                if (response.status === 200) {
                    console.log('good')
                    const gen_puser = response.data.data[0];
                    const m_performance = response.data.data[1];
                    const ctcl_puser = response.data.data[2];
                    const conversion_rate = response.data.data[3];
                    const t_viewer = response.data.data[4];
                    const t_activeuser = response.data.data[5];
                    const t_conversionrate = response.data.data[6];
                    const t_maxprogram = response.data.data[7];
                    console.log(t_viewer);
                    setGen(gen_puser);
                    setCtcl(ctcl_puser);
                    setM_Performance(m_performance);
                    setCr(conversion_rate);
                    setTV(t_viewer);
                    setTAU(t_activeuser);
                    setTCR(t_conversionrate);
                    setTMP(t_maxprogram);
                    
                } else if (response.status !== 200) {
                    console.log("연결 실패");
                }

                // if (!isComponentMounted){
                //     return;
                // }

            } catch (error) {
                console.log("try 실패", error);
            }
        }
        getdata();

        // const currentDate = new Date();

        // // Extract year, month, and day components
        // const year = currentDate.getFullYear();
        // const month = (currentDate.getMonth() + 1).toString().padStart(2, '0'); // Month is 0-indexed
        // const day = currentDate.getDate().toString().padStart(2, '0');

        // // Format the date as a string
        // const formattedDate = `${year}${month}${day}`;

        // console.log(formattedDate)



        // let isComponentMounted = true;


    }, []);
    console.log(t_maxprogram);
    const prepareData = (data) => {
        return data.map((item) => ({
            ...item,
            beginner: parseFloat(item.beginner.toFixed(3)),
            starter: parseFloat(item.starter.toFixed(3)),
            standard: parseFloat(item.standard.toFixed(3)),
            heavy: parseFloat(item.heavy.toFixed(3)),
        }));
    };

    const roundedcr = Math.round(t_conversionrate * 1000) / 1000;

    return (
        <div>
            <div>
                <div className="z-10 !sticky top-[0] overflow-block relative flex bg-red-A400 flex md:flex-col h-[90px] flex-row md:gap-5 items-start justify-end pb-1.5 w-full">
                    <div className="flex items-center justify-center">
                        <Text className="absolute font-yogi font-lighter left-[38%] top-[7%] mt-[1.7%] md:text-3xl text-4xl text-white-A700 tracking-[-0.30px]">
                            HELLO 00 TV - Monitoring
                        </Text>
                    </div>
                </div>
                <div style={{ width: '1000px', margin: '0 auto', marginTop: '3%' }} >
                    <div className="flex flex-col items-center justify-start" style={{ marginTop: '3%', marginLeft: '-4%' }}>
                        <Text
                            className="leading-[10.00px] pl-[50px] ml-[5%] sm:text-[30px] md:text-[50px] text-[25px] text-black-900 tracking-[-0.13px] w-full font-yogi inline-block"
                            size="txtYogi"
                        >
                            <span className="text-black-900 font-yogi">
                                Today's &nbsp; viewer: {t_viewer} &nbsp;&nbsp;  active user: {t_activeuser} &nbsp;&nbsp;  conversion rate: {roundedcr}
                            </span>
                        </Text>
                    </div>
                </div>


                <div style={{ width: '1000px', height: '500px', margin: '0 auto' }}>
                    <div className="flex flex-col items-center justify-start" style={{ marginTop: '3%', marginLeft: '-4%' }}>
                        <Text
                            className="leading-[10.00px] pl-[50px] ml-[5%] sm:text-[30px] md:text-[50px] text-[20px] text-black-900 tracking-[-0.13px] w-full font-yogi inline-block"
                            size="txtYogi"
                        >
                            <span className="text-black-900 font-yogi">
                                Monthly conversion rate
                            </span>
                        </Text>
                    </div>
                    <ResponsiveBar
                        data={prepareData(conversion_rate)}
                        keys={[
                            'beginner',
                            'starter',
                            'standard',
                            'heavy'
                        ]}
                        indexBy='date'
                        width={900}
                        height={500}
                        margin={{ top: 30, right: 110, bottom: 50, left: 110 }}

                        padding={0.2}
                        groupMode='grouped'
                        valueScale={{ type: 'linear' }}
                        indexScale={{ type: 'band', round: true }}
                        colors={{ scheme: 'nivo' }}
                        borderColor='black'
                        fill="red"
                        axisRight={null}
                        axisBottom={{
                            tickSize: 5,
                            tickPadding: 5,
                            tickRotation: 0,
                            legend: 'date',
                            legendPosition: 'middle',
                            legendOffset: 32,
                            truncateTickAt: 0
                        }}
                        axisLeft={{
                            tickSize: 5,
                            tickPadding: 5,
                            tickRotation: 0,
                            legend: 'conversion rate',
                            legendPosition: 'middle',
                            legendOffset: -40,
                            truncateTickAt: 0
                        }}
                        labelSkipWidth={12}
                        labelSkipHeight={9}
                        labelTextColor='black'

                        legends={[
                            {
                                dataFrom: 'keys',
                                anchor: 'bottom-right',
                                direction: 'column',
                                justify: false,
                                translateX: 220,
                                translateY: 0,
                                itemsSpacing: 6,
                                itemWidth: 200,
                                itemHeight: 20,
                                itemDirection: 'left-to-right',
                                itemOpacity: 0.85,
                                symbolSize: 16,
                                effects: [
                                    {
                                        on: 'hover',
                                        style: {
                                            itemOpacity: 1,

                                        }
                                    }
                                ]
                            }
                        ]}
                        role='application'
                        ariaLabel='Nivo bar chart demo'
                        // layers={[
                        //     'grid',
                        //     'axes',
                        //     'bars',
                        //     (layerProps) => {
                        //       const { bars } = layerProps;

                        //       return bars.map((bar) => (
                        //         <g key={bar.id} transform={`translate(${bar.x + bar.width / 2},${bar.y + bar.height / 2})`}>
                        //           <text
                        //             x={0}
                        //             y={10}
                        //             textAnchor="middle"
                        //             dominantBaseline="text-before-edge"
                        //             fill="rgba(255, 255, 255, 0.0)"
                        //             fontSize={11}
                        //             transform="rotate(0)"
                        //           >
                        //             {bar.data.value.toFixed(2)}
                        //           </text>
                        //         </g>
                        //       ));
                        //     },
                        //   ]}
                        barAriaLabel={e => e.id + ": " + e.formattedValue + " in date: " + e.indexValue}

                    />
                </div>

                <div style={{ width: '1000px', height: '500px', margin: '0 auto' }}>
                    <div className="flex flex-col items-center justify-start" style={{ marginTop: '7%', marginLeft: '-4%' }}>
                        <Text
                            className="leading-[10.00px] pl-[50px] ml-[5%] sm:text-[30px] md:text-[50px] text-[20px] text-black-900 tracking-[-0.13px] w-full font-yogi inline-block"
                            size="txtYogi"
                        >
                            <span className="text-black-900 font-yogi">
                                Monthly Model Performance
                            </span>
                        </Text>
                    </div>
                    <ResponsiveLine

                        data={m_performance}
                        width={900}
                        height={500}
                        margin={{ top: 30, right: 110, bottom: 50, left: 110 }}
                        xScale={{ type: 'point' }}
                        yScale={{
                            type: 'linear',
                            min: 'auto',
                            max: 'auto',
                            stacked: true,
                            reverse: false
                        }}
                        yFormat=" >-.2f"
                        curve="cardinal"
                        axisTop={null}
                        axisRight={null}
                        axisBottom={{
                            tickSize: 5,
                            tickPadding: 5,
                            tickRotation: 0,
                            legend: 'Date',
                            legendOffset: 36,
                            legendPosition: 'middle'
                        }}
                        axisLeft={{
                            tickSize: 5,
                            tickPadding: 5,
                            tickRotation: 0,
                            legend: 'Performance',
                            legendOffset: -40,
                            legendPosition: 'middle'
                        }}
                        colors={{ scheme: 'nivo' }}
                        lineWidth={4}
                        pointSize={7}
                        pointColor={{ from: 'color', modifiers: [] }}
                        pointBorderWidth={1}
                        pointBorderColor={{ from: 'serieColor', modifiers: [] }}
                        pointLabelYOffset={-12}
                        crosshairType="bottom"
                        useMesh={true}
                        legends={[
                            {
                                anchor: 'bottom-right',
                                direction: 'column',
                                justify: false,
                                translateX: 62,
                                translateY: 10,
                                itemsSpacing: 2,
                                itemDirection: 'left-to-right',
                                itemWidth: 40,
                                itemHeight: 23,
                                itemOpacity: 0.75,
                                symbolSize: 13,
                                symbolShape: 'circle',
                                symbolBorderColor: 'rgba(0, 0, 0, .5)',
                                effects: [
                                    {
                                        on: 'hover',
                                        style: {
                                            itemBackground: 'rgba(0, 0, 0, .03)',
                                            itemOpacity: 1
                                        }
                                    }
                                ]
                            }
                        ]}

                    />
                </div>

                <div style={{ width: '900px', height: '500px', margin: '0 auto', display: 'flex' }}>
                    <div style={{ width: '50%' }}>
                        <div className="flex flex-col items-center justify-start" style={{ marginTop: '12%', marginLeft: '-15%' }}>
                            <Text
                                className="leading-[10.00px] pl-[50px] ml-[5%] sm:text-[30px] md:text-[50px] text-[20px] text-black-900 tracking-[-0.13px] w-full font-yogi inline-block"
                                size="txtYogi"
                            >
                                <span className="text-black-900 font-yogi">
                                    Weekly Ct_cl chart
                                </span>
                            </Text>
                        </div>
                        <ResponsivePie
                            data={ctcl_puser}
                            width={800}
                            height={300}
                            margin={{ top: 50, right: 465, bottom: 0, left: 0 }}
                            valueFormat=" >-.2f"
                            innerRadius={0.5}
                            padAngle={0.7}
                            cornerRadius={3}
                            activeInnerRadiusOffset={5}
                            activeOuterRadiusOffset={8}
                            colors={{ scheme: 'set3' }}
                            borderWidth={1}
                            borderColor={{ theme: 'background' }}
                            arcLinkLabelsSkipAngle={2}
                            arcLinkLabelsTextOffset={9}
                            //link label color
                            arcLinkLabelsTextColor="#333333"
                            arcLinkLabelsOffset={9}
                            arcLinkLabelsDiagonalLength={0}
                            arcLinkLabelsStraightLength={0}
                            arcLinkLabelsThickness={5}
                            arcLinkLabelsColor={{ from: 'color', modifiers: [] }}
                            arcLabel="value"
                            arcLabelsSkipAngle={10}
                            arcLabelsTextColor="black"
                            legends={[
                                {
                                    anchor: 'right',
                                    direction: 'column',
                                    justify: false,
                                    translateX: 170,
                                    translateY: 50,
                                    itemsSpacing: 0,
                                    itemWidth: 200,
                                    itemHeight: 23,
                                    itemTextColor: '#999',
                                    itemDirection: 'left-to-right',
                                    itemOpacity: 1,
                                    symbolSize: 13,
                                    symbolShape: 'circle',
                                    effects: [
                                        {
                                            on: 'hover',
                                            style: {
                                                itemTextColor: '#000'
                                            }
                                        }
                                    ]
                                }
                            ]}

                        />
                    </div>
                    <div style={{ width: '50%' }}>
                        <div className="flex flex-col items-center justify-start" style={{ marginTop: '12%', marginLeft: '-13%' }}>
                            <Text
                                className="leading-[10.00px] pl-[50px] ml-[5%] sm:text-[30px] md:text-[50px] text-[20px] text-black-900 tracking-[-0.13px] w-full font-yogi inline-block"
                                size="txtYogi"
                            >
                                <span className="text-black-900 font-yogi">
                                    Weekly program genre chart
                                </span>
                            </Text>
                        </div>
                        <ResponsivePie
                            data={gen_puser}
                            width={800}
                            height={300}
                            margin={{ top: 50, right: 0, bottom: 0, left: -430 }}
                            valueFormat=" >-.2f"
                            innerRadius={0.5}
                            padAngle={0.7}
                            cornerRadius={3}
                            activeInnerRadiusOffset={5}
                            activeOuterRadiusOffset={8}
                            colors={{ scheme: 'set3' }}
                            borderWidth={1}
                            borderColor={{ theme: 'background' }}
                            arcLinkLabelsSkipAngle={2}
                            arcLinkLabelsTextOffset={9}
                            //link label color
                            arcLinkLabelsTextColor="#333333"
                            arcLinkLabelsOffset={9}
                            arcLinkLabelsDiagonalLength={0}
                            arcLinkLabelsStraightLength={0}
                            arcLinkLabelsThickness={5}
                            arcLinkLabelsColor={{ from: 'color', modifiers: [] }}
                            arcLabel="value"
                            arcLabelsSkipAngle={10}
                            arcLabelsTextColor="black"
                            legends={[
                                {
                                    anchor: 'right',
                                    direction: 'column',
                                    justify: false,
                                    translateX: -250,
                                    translateY: 35,
                                    itemsSpacing: 0,
                                    itemWidth: 200,
                                    itemHeight: 23,
                                    itemTextColor: '#999',
                                    itemDirection: 'left-to-right',
                                    itemOpacity: 1,
                                    symbolSize: 13,
                                    symbolShape: 'circle',
                                    effects: [
                                        {
                                            on: 'hover',
                                            style: {
                                                itemTextColor: '#000'
                                            }
                                        }
                                    ]
                                }
                            ]}
                        />
                    </div>



                </div>



            </div>
            <Text
                className="leading-[100.00px] pl-[50px] sm:text-[21px] md:text-[23px] text-[10px] text-black-900 tracking-[-0.13px] w-full"
                style={{ backgroundColor: 'gray', width: '100%', height: '150px' }}
            >
                <span className="text-black-900 font-yogi mt-[-2%] pt-[1%]" style={{ display: 'block', marginBottom: '8px' }}>
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
    )
}

export default CP;