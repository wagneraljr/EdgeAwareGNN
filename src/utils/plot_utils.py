from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


class PlotUtils:
    @staticmethod
    def plot_metrics(metrics_dict,path_to_save: str, filename):
        labels = list(metrics_dict.keys())
        mae_scores = [metrics['MAE'] for metrics in metrics_dict.values()]
        rmse_scores = [metrics['RMSE'] for metrics in metrics_dict.values()]
        
        x = np.arange(len(labels))  # Localização das etiquetas
        width = 0.3  # Largura das barras
        
        fig, ax = plt.subplots(figsize=(15, 7))
        
        rects1 = ax.bar(x - width/2, mae_scores, width, label='MAE')
        rects2 = ax.bar(x + width/2, rmse_scores, width, label='RMSE')

        ax.set_ylabel('Scores')
        ax.set_title('Scores by model')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        fig.tight_layout()

        plt.savefig(path_to_save + filename)

    @staticmethod
    def plot_multi_r2(target,multi_predictions:list[list],model_names:list[str], path_to_save:str, filename:str):
        # Calculando e imprimindo as métricas R2 para cada modelo
        r2s = []
        for prediction in multi_predictions:
            r2s.append(r2_score(target,prediction))
        # Plotando a pontuação R2
        plt.figure(figsize=(10, 5))
        plt.rc('font', size=20)
        plt.bar(model_names, r2s)
        plt.ylabel('R2')
        plt.savefig(path_to_save + filename)
        return r2s
    
    @staticmethod
    def plot_predicted_vs_actual(predictions_dict, target_values, path_to_save:str, filename:str):
        plt.figure(figsize=(15, 6))
        plt.rc('font', size=15)
        for idx, (model_name, predicted_values) in enumerate(predictions_dict.items(), 1):
            plt.subplot(1, len(predictions_dict), idx)
            plt.scatter(target_values, predicted_values, alpha=0.5)
            plt.plot([min(target_values), max(target_values)], [min(target_values), max(target_values)], 'r--')
            plt.title(f'{model_name}')
            plt.xlabel('Valores Reais')
            plt.ylabel('Valores Previstos')
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(path_to_save + filename)

    @staticmethod
    def plot_loss_curves(losses_dict, markers,  path_to_save:str, filename:str):
        plt.figure(figsize=(10, 6))
        plt.rc('font', size=20)
        for model_name, loss_values in zip(losses_dict.keys(), losses_dict.values()):
            plt.plot(loss_values, label=model_name, marker=markers[model_name])
        plt.xlabel('Épocas')
        plt.ylabel('Perda (MSE)')
        plt.xlim(0, 60)
        plt.legend()
        plt.grid(True)
        plt.savefig(path_to_save + filename)

    