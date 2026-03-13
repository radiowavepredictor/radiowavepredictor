from pydantic import BaseModel,field_validator
import numpy as np

# los,nlos共通のパラメータ
class FadingBase(BaseModel):
    data_num:int
    c: float  # 光速
    f: float  # 周波数
    delta_d: float

    class Config:
        frozen = True  

    @property
    def lambda_0(self) -> float: # 波長
        return self.c / self.f   

class NLosConfig(FadingBase):
    l: int
   
class LosConfig(FadingBase):
    target_k_db: float
    
    @property
    def r0(self):
        K_linear = 10 ** (self.target_k_db / 10.0)
        r0 = np.sqrt(K_linear)
        return r0
        
class RiceConfig(NLosConfig,LosConfig):
    
    #NOTE インスタンスを作るときに入力する到来波数Lは(los(直接波)の数=1 + Nlosの数)
    #この構造体が持つLはNlosの波の数なので、入力-1する(直接波の数分減らす)
    @field_validator("l", mode="before")
    @classmethod
    def l_minus_1(cls, v):
        return v - 1