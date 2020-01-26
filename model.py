import numpy as np
import pandas as pd
import numpy.linalg as linalg
import matplotlib.pyplot as plt

from data_preparation import get_rates, get_decomp


instruments=['usd','rub','fx']
def stoch_wrapper(decomp):
    def make_stoch(num):
        sigma=[0.03, 0.0093, 0.11]
        stoch_generator = np.dot(np.random.normal(size=(num,3)),decomp)*sigma
        return stoch_generator
    return make_stoch


stoch_generator = stoch_wrapper(get_decomp())

"""

Идея из чата.

theta(t) = Ft(0,t)+a*F(0,t)+(sigma^2)/2a * (1-exp(-2at))

dXt/Xt = k*(theta - ln Xt)*dt + sigma_fx(t)*dWt




Вопросы.
1. Как все-таки считаются рейты? мы правильно это делаем?
3. Нужно ли что-то под что-то фитить, или просто голые симуляции?


Туду
1. исправить данные:
    рейты обмена - в одном направлении (а не в двух разных)
    порядок инструментов - одинаковый везде
    безкостыльные экстраполированные значения+диффы


"""
def simulate_hull_white(
    sim_number = 10,):
    rub_alpha=0.03
    usd_alpha=0.02
    sigma=[0.03, 0.0093, 0.11]
    k_fx=0.015
    dt=14/365
    timesteps = 26

    (
        curve_rub,
        curve_usd,
        curve_fx,
        curve_rub_df,
        curve_usd_df,
        curve_fx_df,
        init
        ) = get_rates()


    results = np.zeros(shape=(timesteps+1, 3, sim_number))

    passed_time=0

    for sim_ix in range(sim_number):
        results[0,:,sim_ix] = init
        stochs = stoch_generator(timesteps+1)
        for i, (rate_rub, rate_usd, rate_fx,df_rub, df_usd,df_fx, stoch_tuple) in enumerate(zip(curve_rub,curve_usd,curve_fx,curve_rub_df,curve_usd_df, curve_fx_df, stochs)):
            passed_time+=dt

            theta_rub = df_rub + rub_alpha*rate_rub + (sigma[0]**2)*(1-np.exp(-2*rub_alpha*passed_time))/2*rub_alpha
            theta_usd = df_usd + usd_alpha*rate_usd + (sigma[1]**2)*(1-np.exp(-2*usd_alpha*passed_time))/2*usd_alpha

            results[i+1,0,sim_ix] = (theta_rub - rub_alpha* results[:,0,sim_ix].sum())*dt+stoch_tuple[0]
            results[i+1,1,sim_ix] = (theta_usd - usd_alpha* results[:,1,sim_ix].sum())*dt+stoch_tuple[1]
            results[i+1,2,sim_ix] = k_fx*(rate_fx - np.log( results[:,2,sim_ix].sum()))*dt+stoch_tuple[2]

    return results


def perform_simulations_basic_mode(sim_number=1000):
    results = simulate_hull_white(sim_number=sim_number)
    # results - np array of shape(no of timesteps, no of instruments, no of simulations)
    plot_results(results)
    return results

def plot_results(result, show_diffs=True):
    fig,ax = plt.subplots(2,3,figsize=(25,12))
    # if show_diffs:
    titles=['Процентная ставка в рублях, приращения', "Процентная ставка в доларах, приращения", "Обменный курс, приращения"]
    for i in range(3):
        ax[0][i].plot(result[1:,i,:], alpha=0.1)
        ax[0][i].plot(np.quantile(result[1:,i,:], q=0.95, axis=1))
        ax[0][i].set_xlabel('Шаг симуляции')


        # pd.DataFrame(result[1:,i,:].T).quantile(0.95).plot(ax=ax[0][i])
        ax[0][i].set_title(titles[i]+ f'\nVaR 95%\nmax = {round(np.quantile(result[1:,i,:], q=0.95, axis=1).max(), 4)}')
    # else:
    titles=['Процентная ставка в рублях', "Процентная ставка в доларах", "Обменный курс"]
    for i in range(3):
        ax[1][i].plot(result[:,i,:].cumsum(axis=0), alpha=0.1)
        # pd.DataFrame(result[:,i,:].cumsum(axis=0).T).quantile(0.05).plot(ax=ax[1][i])
        ax[1][i].set_title(titles[i])
        ax[1][i].set_xlabel('Шаг симуляции')

    plt.show()


def get_optimal_sim_num():
    sim_nums = np.linspace(100,100000,51)
    quantiles = {}
    raw = {}
    for sim in sim_nums:
        local_result = simulate_hull_white(int(sim))
        max_quantiles = np.quantile(local_result[1:,:,:],q=0.95,axis=2).max(axis=0)
        quantiles[int(sim)]=max_quantiles
        raw[int(sim)]=local_result

    df = pd.DataFrame(quantiles).T#.diff().abs().plot()
    df.rename(columns={0:'RUB',1:'USD',2:'FX'}, inplace=True)
    fig, axes=plt.subplots(ncols=2, figsize=(15,9))
    df.plot(title='Значение 95% VaR приращений',ax=axes[0])
    df.diff().abs().plot(title='Модуль дельты значения 95% VaR приращений',ax=axes[1])
    axes[0].set_ylabel('Значение VaR')
    axes[1].set_ylabel('Отличие VaR при повышении количества итераций')
    axes[0].set_xlabel('Количество симуляций')
    axes[1].set_xlabel('Количество симуляций')
    
    plt.show()
    return quantiles, raw


# import model
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# q, r = model.get_optimal_sim_num()