__author__ = 'pengfeishen','marinab', 'TassianaH', 'J.Artur.B' 'ThalitaO'
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import configparser
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import math
import logging
import datetime
inicio_tempo = datetime.datetime.now()
lista_ano_simulado = []
# Timestep
setup_file = "parametros_quanti_registro_baixo.ini"
setup = configparser.ConfigParser()
setup.read(setup_file)
dt = float(setup['TIMESTEP']['dt'])

Ano_simulado = int(input('Qual o ano simulado?'))
print(Ano_simulado)

#importando dados de chuva
lista_area_telhado = [100, 200, 250]
lista_vol_reservatorio = [1, 2, 3, 5, 10]
lista_todos_dias =[]
lista_Vin_telhado = []
lista_todos_telhados =[]
lista_todos_volumes =[]
lista_todos_telhados_mes =[]
lista_todos_volumes_mes =[]
lista_volume_total_demandado = []
lista_volume_total_demandado_diario = []
n_dias_c_demanda = []
lista_Vtotal_in = []
lista_Vtotal_over = []
lista_Vtotal_et = []
lista_Vtotal_pipe = []
lista_Vstorage_teorico = []
lista_Vstorage_real = []
lista_volume_armazenado_ultimo_dia_ano = []
lista_todos_anos= []
lista_todos_meses= []

for i in range(len(lista_area_telhado)):

    # inserindo coluna com dados de vazão

    dados_de_chuva = pd.read_csv("T:\\Resultados_Capitulo_1_\\Montante\\Reservatorio_a_montante\\hadgem85.dat", sep='\s+',
                                 engine='python')[5259455:5364863] #2056 BISSEXTO
    dados_de_chuva.head()
    mes = (dados_de_chuva['mes'])
    ano = (dados_de_chuva['ano'])
    lista_mes = mes.tolist()
    lista_ano = ano.tolist()
    Q_rain = (dados_de_chuva['prec'])
    Qrain = ((Q_rain * 0) / 1000) / dt
    tQrain = Qrain.tolist()
    Area_telhado = lista_area_telhado[i]
    Qin_file = (((Q_rain * Area_telhado) / 1000) * 0.8) / dt
    Vin_telhado = (((sum(Qin_file)) * dt * 1000))
    dados_de_chuva.insert(loc=7, column='Qin_telhado', value=Qin_file)
    print("\033[1:30m........ Dados Telhado ........ \033[m")
    print('Área do telhado: {:.4f}'.format(Area_telhado))
    print('Volume escoado no telhado {:.4f}'.format(((sum(Qin_file)) * dt * 1000)))  # L

    for j in range(len(lista_vol_reservatorio)):
        volume_reservatorio = lista_vol_reservatorio[j]
        print('Volume Reservatório: {:.4f}'.format(volume_reservatorio))

        # Evapotranspiração
        Emax_file = pd.read_csv('Emax_cada_mes_ano_bissexto.csv')
        tEmax = Emax_file['ET'].tolist()

        ### 2. Dinâmica reservatório
        volume_util = volume_reservatorio

        volume_armazenado_diario_a = 0
        volume_armazenado_diario = []
        volume_diario = []
        volume_biorretencao = []
        lista_volume_entrada = []
        vazao_biorretencao = []
        demanda_1 = 3 * 0.128205
        lista_demanda = []
        lista_precipitacao_mensal = []
        lista_dias = []

        volume_armazenado_diario.append(volume_armazenado_diario_a)

        for mes in range(1, 13):
            indice_mes = np.where(dados_de_chuva["mes"] == mes)
            df_mes = dados_de_chuva.iloc[indice_mes[0][0]:indice_mes[0][-1] + 1, :]
            dias = list(set(df_mes["dia"]))
            anos = list(set(df_mes["ano"]))
            # print(df_mes)
            precipitacao_mensal = sum(df_mes['Qin_telhado'])
            lista_precipitacao_mensal.append(precipitacao_mensal)

            for dia in dias:
                indice_dia = np.where(df_mes['dia'] == dia)
                df_dia = df_mes.iloc[indice_dia[0][0]:indice_dia[0][-1] + 1, :]
                df_dia = df_dia.reset_index()
                lista_mes.append(mes)
                volume_auxiliar = []
                lista_dias.append(dia)

                # Aqui cada dia está dentro de um dataframe de tamanho de df_dia = 288

                for t in range(len(df_dia)):

                    volume_entrada = df_dia["Qin_telhado"][t] * dt
                    lista_volume_entrada.append(volume_entrada)

                    if t == 0:
                        volume_auxiliar.append(volume_armazenado_diario[-1] + df_dia["Qin_telhado"][0] * dt)

                    else:
                        volume_auxiliar.append(df_dia["Qin_telhado"][t] * dt + volume_auxiliar[t - 1])

                    # DETERMINANDO VOLUME QUE VAI PARA BIORRETENÇÃO
                    if volume_auxiliar[t] > volume_util:

                        volume_biorretencao.append(volume_auxiliar[t] - volume_util)
                        vazao_biorretencao.append((volume_auxiliar[t] - volume_util) / dt)
                        volume_auxiliar[t] = volume_util

                    else:
                        volume_biorretencao.append(0)
                        vazao_biorretencao.append(0)

                # if volume_auxiliar[-1] - demanda_1 < volume_minimo:
                if volume_auxiliar[-1] < demanda_1:
                    demanda = 0
                    lista_demanda.append(demanda)

                else:
                    demanda = 0.128205 * 3
                    lista_demanda.append(demanda)

                volume_armazenado_diario.append(volume_auxiliar[-1] - demanda)

                volume_diario.append(sum(df_dia["Qin_telhado"] * dt))

            lista_volume_total_demandado_diario.append(sum(lista_demanda)*1000)
            lista_todos_telhados_mes.append(Area_telhado)
            lista_todos_volumes_mes.append(volume_reservatorio)
            lista_todos_meses.append(mes)
            lista_todos_anos.append(Ano_simulado)



        # print('Volume entrada biorretenção {:.4f}'.format(sum(volume_biorretencao)))
        print("\033[1:30m........ Dados Reservatório ........ \033[m")

        print('Volume total demandado {:.4f}'.format(((sum(lista_demanda)) * 1000)))  # L

        print('Número de dias com volume disponível {:.4f}'.format((sum(lista_demanda)) / demanda_1))

        print('Diferença do que entra Vin_telhado_total e o Demanda_total {:.4f}'.format(
            ((sum(Qin_file)) * dt * 1000) - (sum(lista_demanda)) * 1000))

        demanda_total = (sum(lista_demanda))
        lista_volume_total_demandado.append((demanda_total)*1000)

        numero_dias_demanda = (sum(lista_demanda)) / demanda_1
        n_dias_c_demanda.append(numero_dias_demanda)
        tQin = vazao_biorretencao

        ###  3. Definition of parameters
        # General
        # Ew = float(setup['GENERAL']['Ew'])
        Kc = float(setup['GENERAL']['Kc'])
        Df = float(setup['GENERAL']['Df'])
        Dw = float(setup['GENERAL']['Dw'])
        Dt = float(setup['GENERAL']['Dt'])
        Dg = float(setup['GENERAL']['Dg'])
        L = float(setup['GENERAL']['L'])
        nf = float(setup['GENERAL']['nf'])
        nw = float(setup['GENERAL']['nw'])
        nt = float(setup['GENERAL']['nt'])
        ng = float(setup['GENERAL']['ng'])
        nn = float(setup['GENERAL']['nn'])
        rwv = float(setup['GENERAL']['rwv'])

        # Ponding zone
        Ab = float(setup['PONDING_ZONE']['Ab'])
        Hover = float(setup['PONDING_ZONE']['Hover'])
        Kweir = float(setup['PONDING_ZONE']['Kweir'])
        wWeir = float(setup['PONDING_ZONE']['wWeir'])
        expWeir = float(setup['PONDING_ZONE']['expWeir'])
        Cs = float(setup['PONDING_ZONE']['Cs'])
        Pp = float(setup['PONDING_ZONE']['Pp'])
        flagp = float(setup['PONDING_ZONE']['flagp'])

        # Unsaturated zone
        A = float(setup['UNSATURATED_ZONE']['A'])
        husz_ini = float(setup['UNSATURATED_ZONE']['husz'])
        nusz_ini = float(setup['UNSATURATED_ZONE']['nusz'])
        Ks = float(setup['UNSATURATED_ZONE']['Ks'])
        sh = float(setup['UNSATURATED_ZONE']['sh'])
        sw = float(setup['UNSATURATED_ZONE']['sw'])
        sfc = float(setup['UNSATURATED_ZONE']['sfc'])
        ss = float(setup['UNSATURATED_ZONE']['ss'])
        gama = float(setup['UNSATURATED_ZONE']['gama'])
        Kf = float(setup['UNSATURATED_ZONE']['Kf'])

        # Saturated zone
        Psz = float(setup['SATURATED_ZONE']['Psz'])
        hpipe = float(setup['SATURATED_ZONE']['hpipe'])
        flagsz = float(setup['SATURATED_ZONE']['flagsz'])
        dpipe = float(setup['SATURATED_ZONE']['dpipe'])
        Cd = float(setup['SATURATED_ZONE']['Cd'])
        Apipe = math.pi * (dpipe / (1000 * 2)) ** 2


        ###  4. Definition of functions

        ## Flows in m3/s

        # Total evapotranspiration
        def cQet(sw, sh, ss, Kc, Emax, A, sEST):
            if sEST <= sh:
                Qet = 0.0
            elif sEST <= sw:
                Qet = A * Emax * Kc * (sEST - sh) / (sw - sh)
            elif sEST <= ss:
                Qet = A * Emax * Kc * (sEST - sw) / (ss - sw)
            else:
                Qet = A * Emax * Kc

            Qet = Qet / (dt * 1000)

            return Qet


        # Ponding zone
        # Overflow from weir

        # Qinfp: Infiltration from the pond to the surrounding soil
        def cQinfp(Kf, Ab, A, Cs, Pp, flagp, hpEST):
            if flagp == 1:
                Qinfp = 0
            else:
                Qinfp = Kf * ((Ab - A) + Cs * Pp * hpEST)
            return Qinfp


        # Qpf: Infiltration from the pond to the filter material
        def cQpf(Ks, hpz, husz, A, Ab, dt, s, nusz, Qinfp):
            Qpf = min(Ks * A * (hpz + husz) / husz, hpz * Ab / dt - Qinfp, (1.0 - s) * nusz * husz * A / dt)

            return Qpf


        def cQover(kWeir, wWeir, hpz, Hover, expWeir, Ab, dt, Qin, Qrain):
            Vcheck = hpz * Ab + dt * (Qin + Qrain)
            # Vcheck = hpz * Ab + dt * (Qin + Qrain) - Qpf * dt
            if Vcheck > Hover * Ab:
                Hcheck = Vcheck / Ab
                Qover = min((Hcheck - Hover) * Ab / dt, kWeir * wWeir * (2 * 9.81) ** 0.5 * (Hcheck - Hover) ** expWeir)

            else:
                Qover = 0

            return Qover

        # Unsaturated zone
        # Capilary rise
        def cQhc(A, ss, sfc, Emax, sEST, Kc):
            s2 = sEST  # min(s + (Qpf-Qet-Qfs)*dt/(nusz*A*husz), 1)
            den = sfc - ss
            if den == 0:
                den = 0.000001

            Cr = 4 * Emax * Kc / (2.5 * (den) ** 2)
            if s2 >= ss and s2 <= sfc:
                Qhc = A * Cr * (s2 - ss) * (sfc - s2)
            else:
                Qhc = 0

            b = s2 - ss
            c = sfc - s2

            return Qhc


        # Infiltration from USZ to SZ
        def cQfs(A, Ks, hpz, husz, gama, nusz, dt, sfc, sEST):
            if sEST >= sfc:
                Qfs = min((A * Ks * (hpz + husz) / husz) * sEST ** gama, (sEST - sfc) * nusz * A * husz / dt)
            else:
                Qfs = 0
            return Qfs


        # Saturated zone
        # Infiltration to surrounding soil
        def cQinf_sz(Kf, A, Cs, Psz, flagsz, hszEST):
            if flagsz == 1:  # lined
                Qinf_sz = 0.0
            else:
                Qinf_sz = Kf * (A + Cs * Psz * hszEST)
            return Qinf_sz


        # Underdrain flow
        def cQpipe(hpipe, A, nsz, dt, Qinf_sz, Apipe, hszEST, Cd):
            if hszEST <= hpipe:
                Qpipe = 0

            else:
                Qpipemax = (hszEST - hpipe) * A * nsz / dt - Qinf_sz
                Qpipemax = max(0, Qpipemax)
                Qpipepossible = Cd * Apipe * ((hszEST - hpipe) * 2 * 9.81) ** 0.5
                Qpipepossible = max(0, Qpipepossible)
                Qpipe = min(Qpipemax, Qpipepossible)

            return Qpipe


        # Porosity
        def cnsz(hsz, L, Dt, Dg, nf, nt, ng):
            if hsz > Dt + Dg and hsz <= L:
                nsz = ((ng * Dg + nt * Dt + nf * (hsz - Dg - Dt))) / hsz
            elif hsz > Dg and hsz <= Dg + Dt:
                nsz = (ng * Dg + nt * (hsz - Dg)) / hsz
            else:
                nsz = ng
            return nsz


        def cnusz(husz, hsz, nusz_ini, ng, Dg, Df):
            if hsz < Dg:
                nusz = (nusz_ini * Df + ng * (Dg - hsz)) / husz
            else:
                nusz = nusz_ini
            return nusz


        ###   5. Model routine

        tt = []
        tQover = []
        tQpf = []
        tQinfp = []
        tQfs = []
        tQhc = []
        tQet = []
        tQinf_sz = []
        tQpipe = []
        tQet1 = []
        tQet2 = []

        thp = []
        ts = []
        thsz = []
        thszEST = []
        thusz = []
        tnsz = []
        tnusz = []
        thpEND = []
        tteta_usz = []
        tteta_sz = []

        indice = list(range(0, len(tQrain)))


        def water_flow_model_run():
            hpEND = 0
            sEST = 0
            hszEST = 0

            hpz = 0.0
            husz = husz_ini

            if hpipe > 0:
                hsz = hpipe
            else:
                # hsz = dpipe/1000 + 0.03
                hsz = 0  ##### mudei isso

            nusz = nusz_ini
            nsz = ng
            s = sw

            for t in range(len(tQrain)):
                Qin = tQin[t]
                Qrain = tQrain[t]
                Emax = tEmax[t]

                ###PZ###

                # beginning
                Qover = cQover(Kweir, wWeir, hpEND, Hover, expWeir, Ab, dt, Qin, Qrain)
                hpz = max(hpEND + dt / Ab * (tQrain[t] + Qin - Qover), 0)
                Qinfp = cQinfp(Kf, Ab, A, Cs, Pp, flagp, hpz)
                Qpf = cQpf(Ks, hpz, husz, A, Ab, dt, s, nusz, Qinfp)

                hpEND = max(hpz - dt / Ab * (Qpf + Qinfp), 0)  # end

                ###USZ###
                sEST = max(min(s + Qpf * dt / (nusz * A * husz), 1), 0)

                if hpipe <= 0.03:
                    Qhc = 0  ######### MUDEI AQUI ##########
                else:
                    Qhc = cQhc(A, ss, sfc, Emax, sEST, Kc)

                Qfs = cQfs(A, Ks, hpEND, husz, gama, nusz, dt, sfc, sEST)

                sEST2 = (sEST * nusz * husz + nsz * hsz) / (nusz * husz + nsz * hsz)

                Qet = cQet(sw, sh, ss, Kc, Emax, A, sEST2)

                Qet1 = Qet * (sEST * nusz * husz) / (sEST * nusz * husz + nsz * hsz)

                if hpipe > 0.03:  ###### acrescentei isso
                    Qet2 = Qet - Qet1
                else:
                    Qet2 = 0

                ###SZ###
                hszEST = hsz + dt * (Qfs - Qhc - Qet2) / A / nsz

                Qinf_sz = cQinf_sz(Kf, A, Cs, Psz, flagsz, hszEST)

                Qpipe = cQpipe(hpipe, A, nsz, dt, Qinf_sz, Apipe, hszEST, Cd)

                hsz = hsz + dt * (Qfs - Qhc - Qinf_sz - Qpipe - Qet2) / A / nsz

                if hsz < 0.0001:  ####acrescentei isso
                    hsz = 0

                husz = L - hsz

                # porosity#
                nsz = cnsz(hsz, L, Dt, Dg, nf, nt, ng)
                nusz = cnusz(husz, hsz, nusz_ini, ng, Dg, Df)

                if t == 0:
                    husz_a = husz_ini
                    nusz_a = nusz_ini
                    s_a = sw
                else:
                    husz_a = thusz[t - 1]
                    nusz_a = tnusz[t - 1]
                    s_a = ts[t - 1]

                s = max(min(1.0, (s_a * husz_a * nusz_a * A + dt * (Qpf + Qhc - Qfs - Qet1)) / (A * husz * nusz)), sh)

                # save all results
                tt.append(t)
                tQover.append(Qover)
                tQpf.append(Qpf)
                tQinfp.append(Qinfp)
                tQfs.append(Qfs)
                tQhc.append(Qhc)
                tQet.append(Qet)
                tQinf_sz.append(Qinf_sz)
                tQpipe.append(Qpipe)
                tQet1.append(Qet1)
                tQet2.append(Qet2)

                thp.append(hpz)
                ts.append(s)
                thusz.append(husz)
                thsz.append(hsz)
                thszEST.append(hszEST)
                tnsz.append(nsz)
                tnusz.append(nusz)
                thpEND.append(hpEND)
                tteta_usz.append(s * nusz)
                tteta_sz.append(nsz)



        if __name__ == '__main__':


            water_flow_model_run()

            ###   6. Saving the results in CSV
            dict_data = {
                't': tt,
                'mes': mes,
                'Qrain': tQrain[:len(tt)],
                'Qin': tQin[:len(tt)],
                'Qet': tQet[:len(tt)],
                'Qet_1': tQet1[:len(tt)],
                'Qet_2': tQet2[:len(tt)],
                'hpEND': thpEND[:len(tt)],
                'Qpf': tQpf[:len(tt)],
                'Qover': tQover[:len(tt)],
                'Qfs': tQfs[:len(tt)],
                'Qhc': tQhc[:len(tt)],
                'Qpipe': tQpipe[:len(tt)],
                'teta_usz': tteta_usz[:len(tt)],
                'teta_sz': tteta_sz[:len(tt)],
                'Qinfp': tQinfp[:len(tt)],
                'Qinf_sz': tQinf_sz[:len(tt)],
                'hpz': thp[:len(tt)],
                's': ts[:len(tt)],
                'husz': thusz[:len(tt)],
                'hsz': thsz[:len(tt)],
                'nsz': tnsz[:len(tt)],
                'nusz': tnusz[:len(tt)],
                'hszEST': thszEST[:len(tt)]
            }

            data = pd.DataFrame(dict_data)

            # data[['Qin', 'Qover', 'Qpipe', 'Qinf_sz']].plot(figsize=(9, 5), linewidth=1.5)

            # plt.show()

            ###  6. Water balance

            Qin_total = data['Qin'].sum()

            lista_Qin = data['Qin'].tolist()
            Qrain_total = data['Qrain'].sum()
            lista_Qrain = data['Qrain'].tolist()
            Vtotal_in = Qin_total * dt + Qrain_total * dt
            lista_Vtotal_in.append(Vtotal_in * 1000)

            Qover_total = data['Qover'].sum()
            lista_Qover = data['Qover'].tolist()
            Vtotal_over = Qover_total * dt
            lista_Vtotal_over.append(Vtotal_over * 1000)

            Qet_total = data['Qet'].sum()
            lista_Qet = data['Qet'].tolist()
            Vtotal_et = Qet_total * dt
            lista_Vtotal_et.append(Vtotal_et * 1000)

            #     Qinf_sz_total = data['Qinf_sz'].sum()
            #     Vtotal_inf_sz = Qinf_sz_total * dt

            Qpipe_total = data['Qpipe'].sum()
            lista_Qpipe = data['Qpipe'].tolist()
            Vtotal_pipe = Qpipe_total * dt
            lista_Vtotal_pipe.append(Vtotal_pipe * 1000)

            # Vtotal_prec = (P * Ac)/1000

            # Vtotal_in_2 = Vtotal_prec*C

            Qpeak_over = data['Qover'].max()

            Qpf_total = data['Qpf'].sum()
            Vtotal_pf = Qpf_total * dt

            Qfs_total = data['Qfs'].sum()
            Vtotal_fs = Qfs_total * dt

            Smax = data['s'].max()

            hmax = data['hpz'].max()

            tpeak = data.loc[data['Qover'] == Qpeak_over, 't'].iloc[0]

            ### calculo do balanço de massa teorico ###

            V_storage_teorico = (Vtotal_in - Vtotal_pipe - Vtotal_et - Vtotal_over) * 1000  # litros
            lista_Vstorage_teorico.append(V_storage_teorico)
            ### calculo do balanço de massa real ###

            n = 11  # number of cells
            dz = L / n
            m_usz = round((L - hpipe) / dz)
            # m_sz = n - m_usz
            Vlayer_usz = (Ab * (L - hpipe)) / m_usz  # mudei isso, antes era: Vlayer_usz = (Ab*(Df))/m_usz
            # Vlayer_sz = (Ab * hpipe) / m_sz  # mudei isso, antes era: Vlayer_sz = (Ab*(Dg))/m_sz

            # usz
            usz_tot_vol_inicial = Vlayer_usz * tteta_usz[0] * m_usz
            total_linhas_usz = len(tteta_usz) - 1
            usz_tot_vol_final = Vlayer_usz * tteta_usz[total_linhas_usz] * m_usz

            # sz
            # sz_tot_vol_inicial = Vlayer_sz * tteta_sz[0] * m_sz
            # total_linhas_sz = len(tteta_sz) - 1
            # sz_tot_vol_final = Vlayer_sz * tteta_sz[total_linhas_sz] * m_sz

            V_storage_real = (usz_tot_vol_final - usz_tot_vol_inicial) * 1000
            lista_Vstorage_real.append(V_storage_real)
            # V_storage_real = (usz_tot_vol_final + sz_tot_vol_final - usz_tot_vol_inicial - sz_tot_vol_inicial) * 1000

            verificacao_balanco = (V_storage_teorico - V_storage_real)

            print("\033[1:30m........ Dados Biorretenção ........ \033[m")

            print('Vtotal_in: {:.2f} L'.format(Vtotal_in * 1000))  # L

            print('Vtotal_et: {:.2f} L'.format(Vtotal_et * 1000))  # L

            print('Vtotal_over: {:.2f} L'.format(Vtotal_over * 1000)) #L

            print('Vtotal_pipe: {:.2f} L'.format(Vtotal_pipe * 1000))  # L

            print('.' * 30)

            print("\033[1:30m...... Balanço de massa ...... \033[m")

            print('V_storage_teorico: {:.2f} L'.format(V_storage_teorico))  # L

            print('V_storage_real: {:.2f} L'.format(V_storage_real))  # L

            print('Verificação do balanco: {:.2f} L'.format(verificacao_balanco))  # L

            porcentagem = ((verificacao_balanco * 100) / V_storage_teorico)

            # print(porcentagem)
            if porcentagem <= 3 and porcentagem >= -3:
                print("\033[1:34mBalanço de massa aceitável! {:.4f} % de erro \033[m".format(porcentagem))
            else:
                print("\033[1:31mBalanço de massa não aceitável: {:.4f} % de erro\033[m".format(porcentagem))

            print('.' * 30)

            volume_armazenado_reservatorio = Vin_telhado - ((demanda_total) * 1000) - (Vtotal_in * 1000)
            lista_volume_armazenado_ultimo_dia_ano.append(volume_armazenado_reservatorio)

        #...

        #fim do codigo

        lista_ano_simulado.append(Ano_simulado)
        lista_Vin_telhado.append(Vin_telhado)
        lista_todos_telhados.append(Area_telhado)
        lista_todos_volumes.append(volume_reservatorio)

        #salvar informacoes
dict_infos = {
    'Ano': lista_ano_simulado,
    'Area_telhado': lista_todos_telhados,
    'volume_reservatorio': lista_todos_volumes,
    'Vin_telhado': lista_Vin_telhado,
    'V_demandado': lista_volume_total_demandado,
    'nº dias c/ demanda': n_dias_c_demanda,
    'Vin_biorretencao': lista_Vtotal_in,
    'V_storage_reservatorio': lista_volume_armazenado_ultimo_dia_ano,
    'V_et': lista_Vtotal_et,
    'V_over': lista_Vtotal_over,
    'V_pipe': lista_Vtotal_pipe,
    'V_storage_teorico': lista_Vstorage_teorico,
    'V_storage_real': lista_Vstorage_real
}

dados_do_sistema = pd.DataFrame(dict_infos)
dados_do_sistema.to_csv('infos_sistema_RM_hadgem8.5_2056.csv', index=False, sep=';', decimal=',')

dict_demanda = {
    'Ano': lista_todos_anos,
    'V_mensal': lista_volume_total_demandado_diario,
    'Area_telhado': lista_todos_telhados_mes,
    'volume_reservatorio': lista_todos_volumes_mes,
    'Mês': lista_todos_meses
}
dados_do_sistema = pd.DataFrame(dict_demanda)
dados_do_sistema.to_csv('infos_demanda_mensal_8.5_2056.csv', index=False, sep=';', decimal=',')

fim_tempo = datetime.datetime.now()
print('Elapsed time: ', fim_tempo - inicio_tempo)

