"""
CNN 모델의 추론 관련 기능을 제공하는 모듈입니다.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from torch.utils.data import DataLoader
from tqdm import tqdm

from Misc import utilities as ut

class CNNInference:
    """CNN 모델 추론을 위한 클래스입니다."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        regression_label: Optional[str] = None
    ):
        """
        Args:
            model: 추론에 사용할 CNN 모델
            device: 추론에 사용할 디바이스
            regression_label: 회귀 레이블 (옵션)
        """
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for inference")
            model = nn.DataParallel(model)
        self.model = model.to(device)
        self.device = device
        self.regression_label = regression_label
        self.label_dtype = torch.float if regression_label else torch.long

    def evaluate(
        self,
        dataloaders_dict: Dict[str, DataLoader],
        new_label: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        모델을 평가합니다.

        Args:
            dataloaders_dict: 평가용 데이터로더
            new_label: 새로운 레이블 (옵션)

        Returns:
            각 데이터셋에 대한 평가 메트릭
        """
        print(f"Evaluating model on device: {self.device}")
        
        self.model.eval()
        
        res_dict = {}
        
        for subset in dataloaders_dict.keys():
            running_metrics = self._init_running_metrics()
            
            with torch.no_grad():
                for batch in dataloaders_dict[subset]:
                    inputs = batch["image"].to(self.device, dtype=torch.float)
                    if new_label is not None:
                        labels = torch.full((inputs.shape[0],), new_label, device=self.device, dtype=self.label_dtype)
                    else:
                        labels = batch["label"].to(self.device, dtype=self.label_dtype)
                        
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    self._update_running_metrics(labels, preds, running_metrics)
                    
                    del inputs, labels
            
            num_samples = len(dataloaders_dict[subset].dataset)
            metrics = self._calculate_metrics(num_samples, running_metrics)
            res_dict[subset] = metrics
            
            print(f"Evaluation on {subset}:")
            print(metrics)
            
        return res_dict

    def ensemble_results(
        self,
        model_list: List[nn.Module],
        dataloader: DataLoader,
        freq: str = "day"  # 'day', 'week', 'month', 'quarter', 'year'
    ) -> pd.DataFrame:
        """
        앙상블 모델의 결과를 생성합니다.

        Args:
            model_list: 앙상블할 모델 리스트
            dataloader: 데이터로더
            freq: 예측 주기 ('day', 'week', 'month', 'quarter', 'year')

        Returns:
            앙상블 결과가 담긴 데이터프레임
        """
        print(f"Getting ensemble results on {self.device} with {len(model_list)} models")
        
        df_columns = ["StockID", "ending_date", "up_prob", "ret_val", "MarketCap"]
        df_dtypes = [object, "datetime64[ns]", np.float64, np.float64, np.float64]
        df_list = []
        
        for batch in tqdm(dataloader, desc="Generating predictions"):
            image = batch["image"].to(self.device, dtype=torch.float)
            
            if self.regression_label is None:
                total_prob = torch.zeros(len(image), 2, device=self.device)
            else:
                total_prob = torch.zeros(len(image), 1, device=self.device)
                
            for model in model_list:
                model.eval()
                with torch.no_grad():
                    outputs = model(image)
                    if self.regression_label is None:
                        outputs = nn.Softmax(dim=1)(outputs)
                total_prob += outputs
                
            del image
            
            batch_df = ut.df_empty(df_columns, df_dtypes)
            batch_df["StockID"] = batch["StockID"]
            batch_df["ending_date"] = pd.to_datetime([str(t) for t in batch["ending_date"]])
            batch_df["ret_val"] = np.nan_to_num(batch["ret_val"].numpy()).reshape(-1)
            batch_df["MarketCap"] = np.nan_to_num(batch["MarketCap"].numpy()).reshape(-1)
            
            if self.regression_label is None:
                batch_df["up_prob"] = total_prob[:, 1].cpu()
            else:
                batch_df["up_prob"] = total_prob.flatten().cpu()
                
            df_list.append(batch_df)
            
        df = pd.concat(df_list)
        df["up_prob"] = df["up_prob"] / len(model_list)
        
        # 거래일 기준으로 정렬
        df = df.sort_values(["ending_date", "StockID"])
        df.reset_index(drop=True, inplace=True)
        
        return df

    def load_ensemble_model(
        self,
        model_paths: List[str],
        ensem: int
    ) -> Optional[List[nn.Module]]:
        """
        앙상블 모델을 로드합니다.

        Args:
            model_paths: 모델 체크포인트 경로 리스트
            ensem: 앙상블 수

        Returns:
            로드된 모델 리스트 또는 None (로드 실패 시)
        """
        model_list = [self.model for _ in range(ensem)]
        
        try:
            for i in range(ensem):
                state_dict = self._load_model_state_dict(model_paths[i])
                model_list[i].load_state_dict(state_dict)
                
                if torch.cuda.device_count() > 1:
                    model_list[i] = nn.DataParallel(model_list[i])
                    
                model_list[i].to(self.device)
                
        except FileNotFoundError:
            print("Failed to load pretrained models")
            return None
            
        return model_list

    @staticmethod
    def _init_running_metrics() -> Dict[str, int]:
        """초기 메트릭 딕셔너리를 생성합니다."""
        return {
            "correct": 0,
            "total": 0,
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0
        }

    @staticmethod
    def _update_running_metrics(
        labels: torch.Tensor,
        preds: torch.Tensor,
        metrics: Dict[str, int]
    ) -> None:
        """
        실행 중인 메트릭을 업데이트합니다.

        Args:
            labels: 정답 레이블
            preds: 예측값
            metrics: 업데이트할 메트릭 딕셔너리
        """
        metrics["correct"] += (preds == labels).sum().item()
        metrics["total"] += len(labels)
        metrics["TP"] += (preds * labels).sum().item()
        metrics["TN"] += ((preds - 1) * (labels - 1)).sum().item()
        metrics["FP"] += (preds * (labels - 1)).sum().abs().item()
        metrics["FN"] += ((preds - 1) * labels).sum().abs().item()

    @staticmethod
    def _calculate_metrics(
        num_samples: int,
        metrics: Dict[str, int]
    ) -> Dict[str, float]:
        """
        최종 메트릭을 계산합니다.

        Args:
            num_samples: 전체 샘플 수
            metrics: 현재까지의 메트릭

        Returns:
            계산된 메트릭 딕셔너리
        """
        TP, TN, FP, FN = metrics["TP"], metrics["TN"], metrics["FP"], metrics["FN"]
        
        return {
            "accuracy": metrics["correct"] / num_samples,
            "diff": ((TP + FP) - (TN + FN)) / num_samples,
            "MCC": (
                float('nan')
                if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) == 0
                else (TP * TN - FP * FN) / 
                np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            )
        }

    def _load_model_state_dict(self, model_path: str) -> Dict[str, torch.Tensor]:
        """
        모델의 state dict를 로드합니다.

        Args:
            model_path: 모델 체크포인트 경로

        Returns:
            모델의 state dict
        """
        print(f"Loading model state dict from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint["model_state_dict"]
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
            
        return new_state_dict