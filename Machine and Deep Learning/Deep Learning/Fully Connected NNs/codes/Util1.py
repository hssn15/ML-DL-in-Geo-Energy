import numpy as np
import pandas as pd
import missingno as msno
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
class DataManipulation:
    """..."""
    def __init__(self, Data):
        self.Data = Data
    

    def MissingDataSummarizer(self):
        """Return a Pandas dataframe describing the contents of a source dataframe including missing values ."""
        
        variables  = []
        dtypes     = []
        count      = []
        unique     = []
        missing    = []
        
        for item in self.Data.columns:
            
            variables.append(item)
            dtypes.append(self.Data[item].dtype)
            count.append(len(self.Data[item]))
            unique.append(len(self.Data[item].unique()))
            missing.append(self.Data[item].isna().sum())

        output = pd.DataFrame({
            'variable': variables, 
            'dtype': dtypes,
            'count': count,
            'unique': unique,
            'missing': missing, 
        })    
        return output
    
    
    def VisualizeMissingData(self, figsize = (20, 15), color = (0, 0, 0), fontsize = 26, sparkline = True, labelsize = 24):
        """Missing Data Visualization by missingno"""
        msno.matrix(self.Data,figsize=figsize,color=color,fontsize=fontsize, sparkline=sparkline);
        plt.tick_params(axis='y', labelsize=labelsize, color='k')
        plt.tick_params(axis='x', labelsize=labelsize,color='k')
        plt.xlabel("Features", fontsize=fontsize,fontname='tahoma')
        plt.ylabel("Number of Samples", fontsize=fontsize,fontname='tahoma')
        plt.tight_layout()
        plt.show()
    
    def DropMissingData(self):
        self.Data = self.Data.dropna(axis=0, how='any')
        return self.Data
    
    def DataAnalysisResults(self,Cov_Matrix = False, Corr = True, Statistics = False, target = 'Water Saturation (SW)'):

        """Using the Input format of pandas DataFrame and output is Dictionary, heat map of correlation coeff and Covariance Matrix of Dataset, uses additional libraries : numpy, scipy and seaborn"""
        
        Columns = np.array(self.Data.columns)
        if Statistics:
            Data_Dict = {}
            for c in Columns:
                aux = {}
            # 1. Maximum:
                aux['Max            '] =   np.max(np.array(self.Data[c]))
            # 2. Minimumn:
                aux['Min            '] =   np.min(np.array(self.Data[c]))
            ## 3.1. Arithmetic:
                aux['Mean_Arth      '] =   np.mean(np.array(self.Data[c]))
            ## 3.2. Geometric: (Due to Overflow Mapping from number to log then from log to number is used ) (Only used for Positive numbers)
                aux['Mean_Geom      '] =   np.exp(np.sum(np.log(np.array(self.Data[c])[np.where(np.array(self.Data[c])>0)]))/len(np.array(self.Data[c])))
            ## 3.3. Harmonic Means: (Only used for Positive numbers)
                aux['Mean_Harm      '] =   len(np.array(self.Data[c])[np.where(np.array(self.Data[c])>0)])/np.sum(1/np.array(self.Data[c])[np.where(np.array(self.Data[c])>0)])
            # 4. Mode:
                aux['Mode           '] =   scipy.stats.mode(np.array(self.Data[c]))
            # 5. Range:
                aux['Range          '] =   np.max(np.array(self.Data[c])) - np.min(np.array(self.Data[c]))
            # 6. Mid-Range: 
                aux['Mid_Range      '] =   (np.max(np.array(self.Data[c])) + np.min(np.array(self.Data[c])))/2
            # 7. Variance:
                aux['Variance       '] =   np.sum((np.array(self.Data[c])-np.mean(np.array(self.Data[c])))**2)/len(np.array(self.Data[c]))
            # 8. IQR
                aux['Q1'             ] =   np.quantile(np.array(self.Data[c]), 0.25)
                aux['Q3'             ] =   np.quantile(np.array(self.Data[c]), 0.75)
                aux['IQR            '] =   aux['Q3'             ] - aux['Q1'             ]
            # 9. Standard Deviation:
                aux['Stand_Dev      '] =   np.sqrt(np.sum((np.array(self.Data[c])-np.mean(np.array(self.Data[c])))**2)/len(np.array(self.Data[c])))
            # 10. Skewness:
                aux['Skewness       '] =   self.Data[c].skew()
            # 11. Kurtosis:
                aux['Kurtosis       '] =   self.Data[c].kurt()
            # 13. Coefficient of Variation: CV = STD/MEAN
                aux['Coeff_Variation'] =   (np.sqrt(np.sum((np.array(self.Data[c])-np.mean(np.array(self.Data[c])))**2)/len(np.array(self.Data[c]))))/np.mean(np.array(self.Data[c]))
            
                Data_Dict[c] = aux
            print(pd.DataFrame(Data_Dict))

    # 12. Covariance:
        if Cov_Matrix:
            print(pd.DataFrame(self.Data.cov()))

    # 14. Correlation Coefficient Analysis:
        if Corr:
            fig, ax = plt.subplots(3, figsize=(20, 9))
            corr1 = self.Data.corr('pearson')[[target]].sort_values(by=target, ascending=False)
            corr2 = self.Data.corr('spearman')[[target]].sort_values(by=target, ascending=False)
            corr3 = self.Data.corr('kendall')[[target]].sort_values(by=target, ascending=False)
            sns.heatmap(corr1, ax=ax[0], annot=True, vmin= -1, vmax = 1).set(title='pearson correlation')
            sns.heatmap(corr2, ax=ax[1], annot=True, vmin= -1, vmax = 1).set(title='spearman correlation')
            sns.heatmap(corr3, ax=ax[2], annot=True, vmin= -1, vmax = 1).set(title='kendall correlation')
            plt.tight_layout(pad=2)

    def DataDistHist(self,figsize = (18, 10),alpha = 0.8, num_bins = 20,subplotsize = [3, 4]):
        fig=plt.figure(figsize=figsize)
        n= 1
        for feature in self.Data.columns:
            plt.subplot(subplotsize[0], subplotsize[1],n)
            plt.hist(self.Data[feature],num_bins,facecolor='blue', alpha=alpha)
            plt.xlabel('{}'.format(feature),fontsize=16,fontname='Tahoma')
            plt.ylabel('Frequency',fontsize=16,fontname='Tahoma')
            n+=1
        fig.tight_layout() 
        plt.show()

    def CrossPlot(self,Target = 'Water Saturation (SW)', c = 'Permeability (Perm)',figsize = (25, 10),subplotsize = [3, 4]):
        fig=plt.figure(figsize=figsize)
        n= 1
        for feature in self.Data.columns:
            plt.subplot(subplotsize[0],subplotsize[1],n)
            plt.scatter(x=feature, y=Target, data=self.Data, c=c, cmap='rainbow')
            plt.ylabel(Target, fontsize=14)
            plt.xlabel(feature, fontsize=14)
            plt.colorbar(label=c)
            n+=1
        fig.tight_layout() 
        plt.show()

    def BoxPlot(self, figsize=(20,10)):
        fig, axs = plt.subplots(1, len(self.Data.columns), figsize=figsize)
        for i, ax in enumerate(axs.flat):
            bplot = ax.boxplot(self.Data.iloc[:,i], notch=True, vert=True,patch_artist=True, flierprops = dict(marker = "o", markerfacecolor = "red"))
            for patch, color in  zip(bplot['boxes'], 'red'):
                patch.set_facecolor(color)
            ax.set_title(self.Data.columns[i], fontsize=9, fontweight='bold')
            ax.tick_params(axis='y', labelsize=9)   
        plt.tight_layout()
        plt.show()

    def tukey_outliers(self,data,idx,  alpha= 2.5):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        data = np.array(data)
        IQR = Q3 - Q1
        lower_bound = Q1 - alpha * IQR
        upper_bound = Q3 + alpha * IQR
        outliers = []
        outliers_idx  = []
        for j in range(len(data)):
            if data[j] < lower_bound or data[j] > upper_bound:
                outliers.append(data[j])
                outliers_idx.append(idx[int(j)])
        return [np.array(outliers), np.array(outliers_idx, dtype=np.integer)]
    