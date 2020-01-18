'''

Техническое задание по групповому проекту - Дари:
Шаги:
    1) Построение кривых процентных ставок в зависимости от даты погашения. В комментариях написано, что шаг для симуляций 2 недели, предполагаю, чтобы у нас была максимально гладкая кривая процентных ставок  - Надя
    2) Выписать даты платежей и суммы по деривативам – Леня
    3) По данным IRS и FRA по процентным ставкам нужно интерполировать ставки процентов в дни платежей (NB – для IR Swap – нужна только кривая FRA (тут исходя из названия надо брать IRS, но пока для меня does not make sense, потому что ориентируемся на плавающую ставку – USD LIBOR 3M, возможно я ошибаюсь); для остальных двух нужно интерполировать обе ставки – российскую и американскую на даты погашения - Леня
    4) 
            i) IR Swap - Получив ставку процента на каждую дату платежа (мы полученную из кривой ставку процента называем теперь форвардной, так как мы ее не видим впрямую на рынке, а вывели ее сами=> этот комментарий, чтобы ориентироваться на код QuantLib, они впрямую используют слово forward), подставляем ее в модель Халла-Уайта как mean reversion level, а также USD альфу и сигму, и в итоге на каждую дату платежа получаем нашу плавающую ставку, в случае процентного свопа USD LIBOR 3M. В итоге у нас будет на каждую дату платежа ставка процента и сумма платежа.
            ii) FX Forward – Получив обе ставки процента на дату погашения через Халла-Уайта, получить курс обмена валют по фомуле Ft = S0*(1+rf)/(1+r)
            iii) FX Call – Получив обе ставки Получив обе ставки процента на дату погашения через Халла-Уайта, получить курс обмена валют по фомуле Ft = S0*(1+rf)/(1+r)
    5) По-хорошему надо бы посчитать стоимость наших деривативов, но он вроде убрал это задание
    6) Начинается подбор сигмы и альфы – через ковариационные матрицы (пока не до конца понимаю процесс)
    7) Начинается расчет квантильной метрики. То есть на каждую дату платежа мы оцениваем на 95% уровне насколько полученная по модели процентная ставка дает нам потери, считаем VaR


one_factor
dr(t) = k(theta(t) - alpha(t)*r(t))*dt + sigma(t) *dW(t)

two_factor

df(r(t))) = (theta(t) - alpha(t)*r(t))*dt + sigma(t)*dW(t)


'''

def simulate_hull_white(frate, date):
    mean_rev_lvl = frate
    alpha=None
    sigma=None



def project():
    do 1
    do 2 
    do 3


def simulation(*args):
    #do_simulation
    #calculate
    return result
import pandas as pd


payment_grafik_fix = pd.Series(zip(
    index=pd.DatetimeIndex(),
    data=payments_fix
))

payment_grafik_float = pd.Series(zip(
    index=pd.DatetimeIndex(),
    data=payments_float
))




usd_fra_rate=[
    2.84253062153074,
    2.80778295018625,
    2.79476910173436,
    2.78672186079151,
    2.78148932826597,
    2.76474857862602,
    2.74385752617657,
    2.72299331906221,
    2.707509663128,
    2.69278142493567,
    2.68063008005223,
    2.67150302574135,
    2.66686195403801,
    2.66005415952352,
    2.67045367500946,
    2.6849951212979,
    2.70525984530401,
    2.72994096813703,
    2.7561976926686,
    2.78172399000649,
    2.80467300483464,
    2.82275184497611,
    2.85791400379487,
    2.87560338050171,
    2.86786144065544,
    2.85316360598699
]
usd_fra_maturity=[
    '3M',
    '3m-6m',
    '6m-9m',
    '9m-12m',
    '12m-15m',
    '15m-18m',
    '18m-21m',
    '21m-24m',
    '24m-27m',
    '27m-30m',
    '30m-33m',
    '33m-36m',
    '36m-39m',
    '4Y',
    '5Y',
    '6Y',
    '7Y',
    '8Y',
    '9Y',
    '10Y',
    '11Y',
    '12Y',
    '15Y',
    '20Y',
    '25Y',
    '30Y',
    ]


rub_irs_rate=[
    8.51987756984622,
    8.6861131906741,
    8.9046038309591,
    9.04938647479667,
    9.16620517437691,
    9.16277248642859,
    9.11912800445127,
    9.06260294132198,
    9.01015779101996,
    8.95942704182927,
    8.91830151124885,
    8.87442664534946,
    8.83969915462692,
    8.836240602553,
    8.79850222938656
]


rate_curve_hypothetical=pd.Series(...)
#want to theck this hypothethis
#we check it with simulation
"""
Положим, мы провели интерполяцию, и на каждый таймстеп симуляции (которая меньше по периодам, чем оригинальные рейты) у нас есть теоретическое значение. его мы вставляем в качестве мин ревержна.
для симуляций сигма постоянная
"""

one_factor
dr(t) = (theta(t) - alpha(t)*r(t))*dt + sigma(t) *dW(t)





def simulate_hull_white(
        curve_rub, 
        curve_us, 
        decomp,#decomposed matrix of covariation
        init #init values for rub rate, usd rate, fx rate
    ):
    theta_fx=1
    k_fx=1
    rub_alpha=1
    usd_alpha=1


    dt=14/365
    
    sim_number = 1000
    timesteps = 25
    results = np.zeros(size=(timesteps, 3, sim_number))


    for sim_ix in range(sim_number):
        results[0,:,sim_ix] = init

        stoch_generator = np.random.normal(scale=decomp, scale=(timesteps,3))
        stoch_generator = decomp*stoch_generator
        
        for i, (rate_rub, rate_usd, stoch_tuple) in enumerate(zip(curve_rub,curve_us, stoch_generator)):
            results[i+1,0,sim_ix] = results[i+1,0,sim_ix] + (rate_rub - rub_alpha*results[i,0,sim_ix])*dt+stoch_tuple[0]
            results[i+1,1,sim_ix] = results[i+1,1,sim_ix] + (rate_usd - usd_alpha*results[i,1,sim_ix])*dt+stoch_tuple[1]
            results[i+1,2,sim_ix] = results[i+1,2,sim_ix] + k_fx*(theta_fx - np.log(results[i,2,sim_ix]))*dt+stoch_tuple[2]

    return results

    
#     dr(t) = k(theta(t) - alpha(t)*r(t))*dt + sigma(t) *dW(t)

# dXt/Xt = k(theta-np.log(Xt))dt + sigma(t)dWt #fx rate 





def main():
    risk_factory_history_1_year = np.array()#shape - (27,3)
    risk_factory_history_1_year_diff=risk_factory_history_1_year.diff()
    cov_matx = np.cov(risk_factory_history_1_year_diff)
    decomposed = linalg.cholesky(cov_matx)






    hypothethical_curve = get_hypo_curve()#series of dates+values
    hypothethical_curve = get_hypo_curve()#series of dates+values


theta = mean_rev


"""
1. матрица корреляций для риск-факторов (процентная ставка ру, us, обменный курс); ежеднеыные данные с maturity=3М
матрица корреляций строится для приращений
берем данные за год, изменение за 2 недели


2. Раскладываем матрицу корреляций по холецкому

3. пихнули в холла-уайта. в качестве исходных позиций - положение рынка на 1.1.2019, параметры - из задания,
для обменного курса - модификация (слайды!)

4. Провели большую пошаговую симуляцию

5. посчитали вар

"""