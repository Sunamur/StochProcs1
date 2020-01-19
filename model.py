import numpy as np
import pandas as pd
import numpy.linalg as linalg
import matplotlib.pyplot as plt

from data_preparation import get_rates, get_decomp
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

"""
Положим, мы провели интерполяцию, и на каждый таймстеп симуляции (которая меньше по периодам, чем оригинальные рейты) у нас есть теоретическое значение. его мы вставляем в качестве мин ревержна.
для симуляций сигма постоянная
"""






# def main():
#     risk_factory_history_1_year = np.array()#shape - (27,3)
#     risk_factory_history_1_year_diff=risk_factory_history_1_year.diff()
#     cov_matx = np.cov(risk_factory_history_1_year_diff)
#     decomposed = linalg.cholesky(cov_matx)



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





def simulate_hull_white(
    curve_rub, 
    curve_usd, 
    curve_fx,
    decomp,#decomposed matrix of covariation
    init, #init values for rub rate, usd rate, fx rate
    rub_alpha=1,
    usd_alpha=1,
    sim_number = 10,
    ):
    sigma=[0.03, 0.0093, 0.11]

    k_fx=0.015
    dt=14/365
    

    timesteps = 27
    results = np.zeros(shape=(timesteps, 3, sim_number))


    for sim_ix in range(sim_number):
        results[0,:,sim_ix] = init

        stoch_generator = np.dot(np.random.normal(size=(timesteps,3)),decomp.values,)*sigma
        
        for i, (rate_rub, rate_usd, rate_fx, stoch_tuple) in enumerate(zip(curve_rub,curve_usd,curve_fx, stoch_generator)):
            results[i+1,0,sim_ix] = results[i,0,sim_ix] + (rate_rub - rub_alpha*results[i,0,sim_ix])*dt+stoch_tuple[0]
            results[i+1,1,sim_ix] = results[i,1,sim_ix] + (rate_usd - usd_alpha*results[i,1,sim_ix])*dt+stoch_tuple[1]
            results[i+1,2,sim_ix] = results[i,2,sim_ix] + k_fx*(rate_fx - np.log(results[i,2,sim_ix]))*dt+stoch_tuple[2]

    return results


def main():

    curve_usd, curve_rub, curve_fx, init = get_rates()


    decomp=get_decomp()
    results = simulate_hull_white(
        curve_rub=curve_rub, 
        curve_usd=curve_usd, 
        curve_fx=curve_fx,
        decomp=decomp,
        init=init,
        rub_alpha=1,
        usd_alpha=1,
        sim_number=10
        )
    # results - np array of shape(no of timesteps, no of instruments, no of simulations)
    plot_results(results)
    return results

def plot_results(result):
    fig,ax = plt.subplots(1,3,figsize=(25,5))
    titles=['Процентная ставка в рублях', "Процентная ставка в доларах", "Обменный курс"]
    for i in range(3):
        ax[i].plot(result[:,i,:], alpha=0.1)
        pd.DataFrame(result[:,i,:].T).quantile(0.05).plot(ax=ax[i])
        ax[i].set_title(titles[i]+ ' (VaR 5%)')
    plt.show()


sigma=[0.03, 0.0093, 0.11]

k_fx=0.015
dt=14/365

sim_number = 10
timesteps = 27
curve_usd, curve_rub, curve_fx, init = get_rates()

dict_of_rates={
    'usd':curve_usd,
    'rub':curve_rub,
    'fx':curve_fx,
}


get_new_rub = lambda prev_state, rate_rub, stoch: prev_state + (rate_rub - rub_alpha*prev_state)*dt+stoch[0]
get_new_usd = lambda prev_state, rate_usd, stoch: prev_state + (rate_usd - usd_alpha*prev_state)*dt+stoch[1]
get_new_fx  = lambda prev_state, rate_fx,  stoch: prev_state + k_fx*(rate_fx - np.log(prev_state))*dt+stoch[2]

dict_of_getters={
    'rub':lambda prev_state, rate_rub, stoch: prev_state + (rate_rub - rub_alpha*prev_state)*dt+stoch[0],
    'usd':lambda prev_state, rate_usd, stoch: prev_state + (rate_usd - usd_alpha*prev_state)*dt+stoch[1],
    'fx':lambda prev_state, rate_fx,  stoch: prev_state + k_fx*(rate_fx - np.log(prev_state))*dt+stoch[2],

}


#id tuple:  rate_type, timestep, value


# def calc_1_step(rate_type, timestep, value, stoch):
#     if rate_type=='usd':
#         getter = get_new_usd

#     return getter(value, dict_of_rates[rate][timestep], stoch)



"""
mini_step = 2 weeks
large_step - parameter (for now = 5 mini steps)
num_steps - calculated parameter (how many large steps)
simulations_number - parameter (for now = 1000)

1. initialize rates.
2. make sim_num simulations of large_step steps; save results
3. bootstrap sim_num points from previous step, make simulations
4. terminate when num_steps is reached

"""

def stoch_wrapper(decomp):
    def make_stoch(num):
        sigma=[0.03, 0.0093, 0.11]
        stoch_generator = np.dot(np.random.normal(size=(num,3)),decomp.values,)*sigma
        return stoch_generator
    return make_stoch


make_stoch = stoch_wrapper(get_decomp())


def make_large_step(
    instrument_kind,
    ministeps, 
    initial_value, 
    initial_timestep):
    history=[(instrument_kind, initial_timestep, initial_value)]
    stoch = make_stoch(ministeps)
    for i in range(ministeps):

        val_increment = dict_of_getters[instrument_kind](
            history[-1][-1], 
            dict_of_rates[instrument_kind][initial_timestep+i], 
            stoch[i])
        history.append(
            ('usd',initial_timestep+i+1, history[-1][-1]+val_increment)
        )
    return history

"""
todo:
dispatcher for large steps
    must initialize, collect results on each checkpoint, bootstrap last results for new large step
"""
