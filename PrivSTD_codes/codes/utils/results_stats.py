import pandas as pd
import time
from typing import List
import matplotlib.pyplot as plt
import os


class ResultStats():
    def __init__(self, config):
        self.DCT_time = 0.0
        self.train_time = 0.0
        self.config = config
        self.losses = []
        self.ortho_losses = []
        self.maes = []
        self.mres = {}
        self.id_mre = {}
        self.hot_spots10mae = []
        self.hot_spots20mae = []
        self.hot_spots40mae = []
        self.hot_spots10reg = []
        self.hot_spots20reg = []
        self.hot_spots40reg = []
        self.forecasting_mae = []
        self.forecasting_mape = []
         # Range Query 相关（新增）
        self.range_maes = []
        self.range_mres = {str(sm): [] for sm in config['test']['sm']}
        self.min_range_mre = {str(sm): float('inf') for sm in config['test']['sm']}
        
        # Identity Range Query（用于对比）
        self.id_range_mae = None
        self.id_range_mre = {}

        for _sm in config['test']['sm']:
            self.mres[str(_sm)] = []
            self.id_mre[str(_sm)] = 0
        self.id_mae = 0

    @property
    def min_mae(self):
        return min(self.maes)

    @property
    def min_mre(self):
        return {k: min(v) for k, v in self.mres.items()}

    def add_DCT_time(self, time):
        self.DCT_time += time

    def add_train_time(self, time):
        self.train_time += time
    def add_loss(self, loss):
        self.losses.append(loss)

    def add_metric(self, ortho_loss):
        self.ortho_losses.append(ortho_loss)
        
    def add_metric(self, ortho_loss):
        self.ortho_losses.append(ortho_loss)

    def add_mae(self, mae):
        self.maes.append(mae)
    
    def add_hotspot(self, hotres):
        self.hot_spots10mae.append(hotres['mae'][10])
        self.hot_spots20mae.append(hotres['mae'][20])
        self.hot_spots40mae.append(hotres['mae'][40])
        self.hot_spots10reg.append(hotres['regret'][10])
        self.hot_spots20reg.append(hotres['regret'][20])
        self.hot_spots40reg.append(hotres['regret'][40])

    def add_forecasting_mae(self, mae):
        self.forecasting_mae.append(mae)

    def add_forecasting_mape(self, mape):
        self.forecasting_mape.append(mape)

    def add_mre(self, mre: List[float]):
        for i, v in enumerate(mre):
            self.mres[str(self.config['test']['sm'][i])].append(v)

    def set_id_mae(self, id_mae):
        self.id_mae = id_mae

    def set_id_mre(self, id_mre: List[float]):
        for i, v in enumerate(id_mre):
            self.id_mre[str(self.config['test']['sm'][i])] = v

    def add_range_mae(self, mae):
        """添加 Range Query MAE"""
        self.range_maes.append(mae)
    
    def add_range_mre(self, mre_dict):
        """
        添加 Range Query MRE
        mre_dict: {'5': 0.123, '10': 0.234, '20': 0.345}
        """
        for sm_key, val in mre_dict.items():
            if sm_key in self.range_mres:
                self.range_mres[sm_key].append(val)
                if val < self.min_range_mre[sm_key]:
                    self.min_range_mre[sm_key] = val

    def write(self, file):
        filedir = '/home/hyj/MyCodes/MultiView/codes/results/'
        filepath = filedir + file
        mre_file_name = self.config['datasets']['name']+'mre.xlsx'
        mre_filepath = os.path.join(filedir, mre_file_name)
        # 判断文件是否存在
        if os.path.exists(filepath):
            print("mre的文件已存在")
            df = pd.read_excel(filepath, sheet_name=self.config['datasets']['name'], engine="openpyxl")
        else:
            # 文件不存在时新建一个空的 DataFrame
            print("文件不存在")
            df = pd.DataFrame()


        # df = pd.read_excel(file, sheet_name=self.config['datasets']['name'], engine='openpyxl')

        self.result = {}
        self.result['model'] = self.config['train']['model']
        self.result['sample_size'] = self.config['datasets']['sample_size']
        self.result['eps'] = self.config['privacy']['eps']
        self.result['img_size'] = self.config['net']['img_size']
        self.result['window_size'] = self.config['net']['window_size']
        self.result['embed_dim'] = self.config['net']['embed_dim']
        self.result['min_mae'] = self.min_mae
        self.result['id_mae'] = self.id_mae
        self.result['min_mae_epoch'] = self.maes.index(self.result['min_mae']) * self.config['train']['eval_freq']
        for k, v in self.id_mre.items():
            self.result['min_mre_' + k] = min(self.mres[k])
            self.result['id_mre_' + k] = v
            self.result['min_mre_' + k + '_epoch'] = self.mres[k].index(self.result['min_mre_' + k]) * \
                                                     self.config['train']['eval_freq']
        for k,v in self.id_range_mre.items():
            self.result['min_range_mre_' + k] = min(self.range_mres[k])
            self.result['id_range_mre_' + k] = v
            self.result['min_range_mre_' + k + '_epoch'] = self.range_mres[k].index(self.result['min_range_mre_' + k]) * \
                                                     self.config['train']['eval_freq']
        self.result['min_hotspot_mae10'] = min(self.hot_spots10mae)
        self.result['min_hotspot_mae10_epoch'] = self.hot_spots10mae.index(self.result['min_hotspot_mae10']) * \
                                                  self.config['train']['eval_freq']
        self.result['min_hotspot_mae20'] = min(self.hot_spots20mae)
        self.result['min_hotspot_mae20_epoch'] = self.hot_spots20mae.index(self.result['min_hotspot_mae20']) * \
                                                  self.config['train']['eval_freq']
        self.result['min_hotspot_mae40'] = min(self.hot_spots40mae)
        self.result['min_hotspot_mae40_epoch'] = self.hot_spots40mae.index(self.result['min_hotspot_mae40']) * \
                                                  self.config['train']['eval_freq']
        self.result['min_forecasting_mae'] = min(self.forecasting_mae)
        self.result['min_forecasting_mae_epoch'] = self.forecasting_mae.index(self.result['min_forecasting_mae']) * \
                                                     self.config['train']['eval_freq']
        self.result['hot_spots10reg'] = min(self.hot_spots10reg)
        self.result['hot_spots10reg_epoch'] = self.hot_spots10reg.index(self.result['hot_spots10reg']) * \
                                                  self.config['train']['eval_freq']
        self.result['hot_spots20reg'] = min(self.hot_spots20reg)
        self.result['hot_spots20reg_epoch'] = self.hot_spots20reg.index(self.result['hot_spots20reg']) * \
                                                  self.config['train']['eval_freq']
        self.result['hot_spots40reg'] = min(self.hot_spots40reg)
        self.result['hot_spots40reg_epoch'] = self.hot_spots40reg.index(self.result['hot_spots40reg']) * \
                                                     self.config['train']['eval_freq']
        self.result['min_forecasting_mape'] = min(self.forecasting_mape)
        self.result['min_forecasting_mape_epoch'] = self.forecasting_mape.index(self.result['min_forecasting_mape']) * \
                                                     self.config['train']['eval_freq']
        
        self.result['train_time'] = self.train_time
        self.result['DCT_time'] = self.DCT_time
        result_df = pd.Series(self.result)

        comment = result_df.copy()
        comment[0] = time.strftime("%Y-%m-%d", time.localtime())
        comment[1] = self.config['train']['comment']
        comment[2:] = ''
        df = df.append(comment, ignore_index=True)

        df = df.append(result_df, ignore_index=True)

        # df.to_excel(file, sheet_name=self.config['datasets']['name'], index=False)
        df.to_excel(filepath, sheet_name=self.config['datasets']['name'], index=False, engine="openpyxl")


                # =========================
        # 4) 新增：把 self.mres['20'] 输出到 mre.xlsx
        # =========================
        mre20_list = self.mres.get('20', None)
        if mre20_list is not None:
            eval_freq = self.config.get('eval_freq', 1)
            mre20_df = pd.DataFrame({
                "epoch": [i * eval_freq for i in range(len(mre20_list))],
                "mre20": mre20_list
            })


            if os.path.exists(mre_filepath):
                with pd.ExcelWriter(mre_filepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                    mre20_df.to_excel(writer, index=False)
            else:
                with pd.ExcelWriter(mre_filepath, engine="openpyxl", mode="w") as writer:
                    mre20_df.to_excel(writer, index=False)


        # plot mre in another figure
        fig1 = plt.figure()
        plt.plot(self.mres['20'])
        print(self.mres['20'])
        fig_file1 = self.config['datasets']['name'] + '_' +self.config['train']['model']+'_'+ str(self.config['privacy']['eps'])+'_'+str(self.config['datasets']['sample_size'])
        fig1.savefig('/home/hyj/MyCodes/MultiView/codes/results/Figure/'+fig_file1+'_mre_20.png')

        # # plot mre in another figure
        # fig2 = plt.figure()
        # plt.plot(self.hot_spots20mae)
        # fig_file2 = self.config['datasets']['name'] + '_' +self.config['train']['model']+'_'+ str(self.config['privacy']['eps'])+'_'+str(self.config['datasets']['sample_size'])
        # fig2.savefig('/home/hyj/MyCodes/MultiView/codes/results/Figure/'+fig_file2+'_hot_spots20mae.png')

        # # plot mre in another figure
        # fig3 = plt.figure()
        # plt.plot(self.hot_spots20reg)
        # fig_file3 = self.config['datasets']['name'] + '_' +self.config['train']['model']+'_'+ str(self.config['privacy']['eps'])+'_'+str(self.config['datasets']['sample_size'])
        # fig3.savefig('/home/hyj/MyCodes/MultiView/codes/results/Figure/'+fig_file3+'_hot_spots20reg.png')         

        # # plot mre in another figure
        # fig4 = plt.figure()
        # plt.plot(self.forecasting_mape)
        # fig_file4 = self.config['datasets']['name'] + '_' +self.config['train']['model']+'_'+ str(self.config['privacy']['eps'])+'_'+str(self.config['datasets']['sample_size'])
        # fig4.savefig('/home/hyj/MyCodes/MultiView/codes/results/Figure/'+fig_file4+'_forecasting_mape.png')

        # fig5 = plt.figure()
        # plt.plot(self.forecasting_mae)
        # fig_file5 = self.config['datasets']['name'] + '_' +self.config['train']['model']+'_'+ str(self.config['privacy']['eps'])+'_'+str(self.config['datasets']['sample_size'])
        # fig5.savefig('/home/hyj/MyCodes/MultiView/codes/results/Figure/'+fig_file5+'_forecasting_mae.png')          
