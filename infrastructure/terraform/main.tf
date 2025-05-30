terraform {
  required_version = ">= 1.0.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.10"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.5"
    }
  }

  backend "s3" {
    # These values should be overridden using a terraform.tfvars file
    bucket         = "mlops-production-system-terraform-state"
    key            = "terraform/state"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "mlops-production-system-terraform-lock"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "MLOps-Production-System"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 18.0"

  cluster_name    = var.cluster_name
  cluster_version = var.kubernetes_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Node groups configuration
  eks_managed_node_groups = {
    main = {
      name = "node-group-1"

      instance_types = ["t3.medium"]
      capacity_type  = "ON_DEMAND"

      min_size     = var.min_nodes
      max_size     = var.max_nodes
      desired_size = var.desired_nodes

      update_config = {
        max_unavailable_percentage = 25
      }
    }
  }

  # Enable OIDC provider for service accounts
  enable_irsa = true

  # Manage aws-auth configmap
  manage_aws_auth_configmap = true
}

# VPC Configuration
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 3.0"

  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true
}

# MLflow deployment with Helm
resource "helm_release" "mlflow" {
  name       = "mlflow"
  repository = "https://larribas.me/helm-charts"
  chart      = "mlflow"
  version    = "1.0.0"
  namespace  = "mlops"
  
  depends_on = [
    module.eks,
    kubernetes_namespace.mlops
  ]

  set {
    name  = "backendStore.postgres.enabled"
    value = "true"
  }

  set {
    name  = "backendStore.postgres.host"
    value = aws_db_instance.mlflow_db.address
  }

  set {
    name  = "defaultArtifactRoot"
    value = "s3://${aws_s3_bucket.mlflow_artifacts.bucket}"
  }
}

# PostgreSQL for MLflow
resource "aws_db_instance" "mlflow_db" {
  identifier             = "${var.cluster_name}-mlflow-db"
  allocated_storage      = 20
  storage_type           = "gp2"
  engine                 = "postgres"
  engine_version         = "13.4"
  instance_class         = "db.t3.micro"
  db_name                = "mlflow"
  username               = "mlflow"
  password               = var.db_password
  skip_final_snapshot    = true
  vpc_security_group_ids = [aws_security_group.db_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.db_subnet_group.name
}

# S3 Bucket for MLflow artifacts
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${var.cluster_name}-mlflow-artifacts"
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts_versioning" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# Create mlops namespace
resource "kubernetes_namespace" "mlops" {
  metadata {
    name = "mlops"
  }
  
  depends_on = [
    module.eks
  ]
}

# Security Group for the database
resource "aws_security_group" "db_sg" {
  name        = "${var.cluster_name}-db-sg"
  description = "Allow database connections from EKS cluster"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description     = "PostgreSQL from EKS"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# DB Subnet Group
resource "aws_db_subnet_group" "db_subnet_group" {
  name       = "${var.cluster_name}-db-subnet-group"
  subnet_ids = module.vpc.private_subnets
}

# Output values
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "kubeconfig_command" {
  description = "Command to generate kubeconfig file for the EKS cluster"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

output "mlflow_url" {
  description = "URL for MLflow UI"
  value       = "kubectl port-forward -n mlops svc/mlflow 5000:5000"
}
