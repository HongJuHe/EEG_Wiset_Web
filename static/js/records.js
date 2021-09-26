// 3차원 데이터 // EEG_Wiset_Web

// value[0]: [['2021_9_08', 1, 0.9164004780000438], ['2021_9_18', 0, 0.9164004780273438], ['2021_9_19', 0, 0.9159774780273438]]
// value[2]: [['depression_insomnia', 'depression_insomnia'], ['sadness_miserable', 'depression_insomnia', 'sadness_miserable']]
// [][][] 구성 3차원 형태

window.onload = function() {
    // Get Record Data
    var data = {{ value|tojson }};
    /*var data = [[['2021_8_08', 1, 0.9164004780000438],
    ['2021_9_08', 1, 0.9164004780000438],
    ['2021_9_18', 0, 0.9164004780273438],
    ['2021_9_19', 0, 0.9159774780273438]],
    [['depression_insomnia', 'depression_insomnia', 'depression_dazed'],
    ['sadness_miserable', 'depression_insomnia', 'sadness_miserable'],
    ['depression_gloomy', 'depression_confidence', 'depression_dazed', 'depression_concentration'],
    ['depression_lethargy', 'depression_lethargy', 'sadness_upset', 'sadness_upset'],
    ['angry_resentment', 'angry_resentment', 'lonely_meaningless', 'sadness_regret']]]*/

    var data_eeg = data[0];
    var data_chat = data[1];

    // Set Graph1 Data
    let today = new Date();
    let year = today.getFullYear() - 1;
    let month = today.getMonth() + 1;

    var dateValues_ = [];
    var dateValues = [];
    var dep0Values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    var dep1Values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    for(let i = 0; i < 12; i++) {
        month++;
        if(month > 12) {
            year++;
            month = 1;
        }
        var str_ = year + '_' + month;
        var str = year + '월 ' + month + '일';
        dateValues_.push(str_);
        dateValues.push(str);
    }
    for(let i = 0; i < data_eeg.length; i++) {
        for(let j = 0; j < dateValues_.length; j++) {
            if(data_eeg[i][0].indexOf(dateValues_[j]) == 0) {
                if(data_eeg[i][1] == 0) {
                    dep0Values[j]++;
                } else {
                    dep1Values[j]++;
                }
            }
        }
    }

    // Set Graph2 Data
    var symptomNames_ = [ 'depression', 'sadness', 'lonely', 'angry', 'emotionaldysregulation']
    var symptomNames_kr = [ '우울함', '슬픔', '외로움', '화', '감정 조절 이상']

    var symptomDepression_ = ['depression', 'depression_gloomy', 'depression_dazed', 'depression_lethargy', 'depression_desire', 'depression_insomnia',
    'depression_interest', 'depression_appetite', 'depression_confidence', 'depression_loser', 'depression_concentration']
    var symptomSadness_ = ['sadness', 'sadness_upset', 'sadness_tear', 'sadness_cry', 'sadness_guilty', 'sadness_miss', 'sadness_remorse',
    'sadness_sad', 'sadness_miserable', 'sadness_vanity', 'sadness_despair', 'sadness_unfair', 'sadness_regret', 'sadness_disappointment']
    var symptomLonely_ = ['lonely', 'lonely_meaningless', 'lonely_suitability', 'lonely_smolder', 'lonely_hard' ]
    var symptomAngry_ = ['angry', 'angry_resentment', 'angry_dissatisfaction', 'angry_hate', 'angry_anger', 'angry_hatred']
    var symptomEmotionaldysregulation_ = [ 'emotionaldysregulation_paralysis', 'emotionaldysregulation_arbitrariness',
        'emotionaldysregulation_suppression', 'emotionaldysregulation_conflict']

    var symptomDepression_kr = ['우울함', '침울함', '멍함', '무기력', '의욕 없음', '불면증',
    '흥미 없음', '식욕 없음', '자존감 하락', '패배감', '집중력 하락']
    var symptomSadness_kr = ['슬픔', '속상', '눈물', '울음', '죄책감', '그리움', '연민',
    '서러움', '비참함', '허망함', '절망', '억울함', '후회', '서운함']
    var symptomLonely_kr = ['외로움', '무의미', '적적함', '울적', '고단함' ]
    var symptomAngry_kr = ['화', '원망', '불만', '미움', '분노', '증오']
    var symptomEmotionaldysregulation_kr = ['감정 마비', '독단', '강압', '갈등']
    var symptomDepression_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    var symptomSadness_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    var symptomLonely_count = [0, 0, 0, 0, 0 ]
    var symptomAngry_count = [0, 0, 0, 0, 0, 0]
    var symptomEmotionaldysregulation_count = [ 0, 0, 0, 0]

    for(let i = 0; i < data_chat.length; i++) {
        for(let j = 0; j < data_chat[i].length; j++) {
            const symptom = data_chat[i][j];
            for(let k = 0; k < symptomNames_.length; k++) {
                if(symptom.indexOf(symptomNames_[k]) == 0) {
                    let num = 0;
                    switch(k) {
                        case 0: // 우울증
                            num = symptomDepression_.indexOf(symptom);
                            symptomDepression_count[num]++;
                            break;
                        case 1: // 슬픔
                            num = symptomSadness_.indexOf(symptom);
                            symptomSadness_count[num]++;
                            break;
                        case 2: // 외로움
                            num = symptomLonely_.indexOf(symptom);
                            symptomLonely_count[num]++;
                            break;
                        case 3: // 분노
                            num = symptomAngry_.indexOf(symptom);
                            symptomAngry_count[num]++;
                            break;
                        case 4: // 감정 조절 장애
                            num = symptomEmotionaldysregulation_.indexOf(symptom);
                            symptomEmotionaldysregulation_count[num]++;
                            break;
                    }

                    break;
                }
            }
        }
    }

    var labels = [], datas = [];
    var MAX_LENGTH = symptomSadness_count.length;
    for(let i = 0; i < MAX_LENGTH; i++) {
        var labels_ = [], datas_ = [];
        labels_.push(symptomDepression_kr[i]);
        labels_.push(symptomSadness_kr[i]);
        labels_.push(symptomLonely_kr[i]);
        labels_.push(symptomAngry_kr[i]);
        labels_.push(symptomEmotionaldysregulation_kr[i])
        labels.push(labels_);
        datas_.push(symptomDepression_count[i]);
        datas_.push(symptomSadness_count[i]);
        datas_.push(symptomLonely_count[i]);
        datas_.push(symptomAngry_count[i]);
        datas_.push(symptomEmotionaldysregulation_count[i]);
        datas.push(datas_);
    }
    var colors = [];
    for(let i = 0; i < MAX_LENGTH; i++) {
        var saturation1 = Math.round(255 / MAX_LENGTH * i);
        colors.push("rgba(" + saturation1 + ", 201, 14, 0.5)");
    }

    var datasets = []
    for(let i = 0; i < MAX_LENGTH; i++) {
        var dataset = { label: labels[i],
            backgroundColor: colors[i],
            borderColor: "rgba(255, 255, 255, 1.0)",
            borderWidth: 2,
            data: datas[i]};
        datasets.push(dataset);
    }

    // Graph1
    new Chart("graph1", {
        type: 'bar',
        data: {
            labels: dateValues,
            datasets: [{
                label: '우울증 아님',
                backgroundColor: "rgba(0, 201, 14, 0.5)",
                data: dep0Values,
            }, {
                label: '우울증',
                backgroundColor: "rgba(150, 201, 14, 0.5)",
                data: dep1Values,
            },]
        },
        options: {
            plugins: {
                datalabels: {
                    color: 'white',
                    display: function(context) {
                        return context.dataset.data[context.dataIndex] > 15;
                    },
                    font: {
                        weight: 'bold'
                    },
                    formatter: Math.round
                }
            },
            aspectRatio: 5 / 3,
            layout: {
                padding: {
                    top: 24,
                    right: 16,
                    bottom: 0,
                    left: 8
                }
            },
            elements: {
                line: {
                    fill: false
                },
                point: {
                    hoverRadius: 7,
                    radius: 5
                }
            },
            scales: {
                xAxes: [{
                    stacked: true,
                    stepSize: 1,
                    min: 0,
                }],
                yAxes: [{ stacked: true }]
            },
            legend: { display: true },
            title: {
                display: true,
                fontSize: 20,
                text: "최근 1년 뇌파 측정 및 우울증 결과 변화"
            }
        }
    });

    // Graph2
    new Chart("graph2", {
        type: 'bar',
        data: {
            labels: symptomNames_kr,
            datasets: datasets
        },
        options: {
            plugins: {
                datalabels: {
                    color: 'white',
                    display: function(context) {
                        return context.dataset.data[context.dataIndex] > 15;
                    },
                    font: {
                        weight: 'bold'
                    },
                    formatter: Math.round
                }
            },
            aspectRatio: 5 / 3,
            layout: {
                padding: {
                    top: 24,
                    right: 16,
                    bottom: 0,
                    left: 8
                }
            },
            elements: {
                line: {
                    fill: false
                },
                point: {
                    hoverRadius: 7,
                    radius: 5
                }
            },
            scales: {
                xAxes: [{
                    stacked: true,
                    stepSize: 1,
                    min: 0,
                }],
                yAxes: [{ stacked: true }]
            },
            legend: { display: false },
            title: {
                display: true,
                fontSize: 20,
                text: "최근 1년 챗봇 대화 정신 증상 변화"
            }
        }
    });


}