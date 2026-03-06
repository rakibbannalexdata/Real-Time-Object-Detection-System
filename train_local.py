import argparse
import logging
import sys
import os
from fastapi import HTTPException
from app.services.training_service import TrainingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 segmentation training locally.")
    parser.add_argument("--project", type=str, required=True, help="Project name (folder in datasets/)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--no-val", action="store_false", dest="val", help="Skip validation after training")
    parser.set_defaults(val=True)
    
    args = parser.parse_args()

    # Ensure project root is in PYTHONPATH so 'app' can be found if run directly
    sys.path.append(os.getcwd())

    service = TrainingService()
    
    try:
        logger.info(f"Starting local training for project: {args.project}")
        logger.info(f"Epochs: {args.epochs}, Image Size: {args.imgsz}, Batch: {args.batch}, Validation: {args.val}")
        
        best_weights_path = service.start_training_sync(
            project_name=args.project,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            val=args.val
        )
        
        logger.info(f"Training completed! Best weights saved at: {best_weights_path}")
        
    except HTTPException as e:
        logger.error(f"Dataset Validation/Service Error: {e.detail}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user. Partial results may be available in the models/ directory.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Training failed due to unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
