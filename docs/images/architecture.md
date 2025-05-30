```mermaid
graph TD
    %% Main title and styles
    classDef pipeline fill:#f0f6ff,stroke:#3273dc,color:#3273dc,stroke-width:2px
    classDef component fill:#ffffff,stroke:#209cee,color:#209cee,stroke-width:1.5px
    classDef note fill:#fffaeb,stroke:#ffdd57,color:#946c00,stroke-width:1px,stroke-dasharray: 5 5
    classDef infra fill:rgba(0,209,178,0.1),stroke:#00d1b2,color:#00d1b2,stroke-width:1.5px,stroke-dasharray: 5 5
    
    %% Infrastructure
    subgraph K8S["Kubernetes Cluster"]
        %% Data Pipeline
        subgraph DP["Data Pipeline"]
            DI[Data Ingestion] :::component
            DV[Data Validation] :::component
            FE[Feature Engineering] :::component
            FSN[Feature Store Integration] :::note
            
            DI --> DV
            DV --> FE
        end
        
        %% Model Training
        subgraph MT["Model Training"]
            ET[Experiment Tracking - MLflow] :::component
            DT[Distributed Training] :::component
            ME[Model Evaluation] :::component
            ABN[A/B Testing Framework] :::note
            
            ET --> DT
            DT --> ME
        end
        
        %% Model Registry
        subgraph MR["Model Registry"]
            MV[Model Versioning] :::component
            MS[Metadata Storage] :::component
            MCI[CI/CD Integration] :::note
            
            MV --> MS
        end
        
        %% API Layer
        subgraph API["API Layer"]
            FA[FastAPI Application] :::component
            PE[Prediction Endpoints] :::component
            HM[Health & Metadata APIs] :::component
            HPA[Horizontal Pod Autoscaling] :::note
            
            FA --> PE
            FA --> HM
        end
        
        %% Monitoring
        subgraph MON["Monitoring"]
            PM[Prometheus Metrics] :::component
            GD[Grafana Dashboards] :::component
            DD[Feature-level Drift Detection] :::component
            RT[Automated Retraining Triggers] :::component
            AM[Alert Manager Integration] :::note
            
            MPT[Model Performance Tracking] :::component
            DQM[Data Quality Monitoring] :::component
            ABT[A/B Testing Analytics] :::component
            LA[Log Aggregation] :::component
            DT2[Distributed Tracing] :::note
            
            PM --> GD
            PM --> DD
            DD --> RT
            MPT --> DQM
            DQM --> ABT
            ABT --> LA
        end
        
        %% Component relationships
        DP -->|Training Data| MT
        DP -->|Metadata| MR
        MT -->|Model Artifacts| MR
        MR -->|Latest Model| API
        API -->|Metrics| MON
        MT -->|Performance Metrics| MON
    end
    
    %% CI/CD Pipeline
    CICD[CI/CD Pipeline: GitHub Actions] :::infra
    CICD -->|Deploy| K8S
    
    %% Apply classes
    class DP,MT,MR,API,MON pipeline
```
