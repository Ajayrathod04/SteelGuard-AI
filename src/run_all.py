import time
from utils import SimpleLogger
from run_eda import run_eda
from train import run_training
from ensemble import run_ensembling
from predict import run_predictions

def main():
    logger = SimpleLogger("SteelGuard-AI-Pipeline")
    start_time = time.time()
    
    logger.info("==================================================")
    logger.info("      STEELGUARD AI PIPELINE ORCHESTRATION        ")
    logger.info("==================================================")
    
    # 1. EDA Phase
    logger.info("--- PHASE 1: EXPLORATORY DATA ANALYSIS ---")
    run_eda()
    # 2. Model Training Phase
    logger.info("--- PHASE 2: MODEL TRAINING (DYNAMIC CV SPLITS) ---")
    run_training()
    
    # 3. Ensembling Phase
    logger.info("--- PHASE 3: WEIGHTED BLEND & THRESHOLD OPTIMIZATION ---")
    run_ensembling()
    
    # 4. Predictions Phase
    logger.info("--- PHASE 4: TEST INFERENCE & SUBMISSION ---")
    run_predictions()
    
    elapsed = time.time() - start_time
    logger.success("==================================================")
    logger.success(f"  ALL PIPELINE PHASES EXECUTED SUCCESSFULLY in {elapsed/60:.2f} mins! ")
    logger.success("==================================================")

if __name__ == '__main__':
    main()
