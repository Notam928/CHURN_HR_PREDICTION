import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Prédiction du Churn - RH",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 30px;
        border-radius: 15px;d
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Initialisation de la session state
if 'page' not in st.session_state:
    st.session_state.page = "🎯 Objectif"
if 'model' not in st.session_state:
    st.session_state.model = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_encoded' not in st.session_state:
    st.session_state.df_encoded = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = None

# Fonction pour charger et préparer les données
@st.cache_data
def load_and_prepare_data():
    try:
        # Charger les données
        df_original = pd.read_excel('HR Data.xlsx')
        df = df_original.copy()
        
        # Nettoyage
        df = df.drop(columns=['EmployeeNumber'])
        df = df.drop(columns=['JobLevel', 'YearsInCurrentRole', 'PerformanceRating'])
        
        # Encodage
        features_to_encode = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 
                              'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
        
        label_encoders = {}
        df_encoded = df.copy()
        
        for feature in features_to_encode:
            le = LabelEncoder()
            df_encoded[feature] = le.fit_transform(df[feature])
            label_encoders[feature] = le
        
        # Séparation X et y
        y = df_encoded['Attrition']
        X = df_encoded.drop(columns=['Attrition'])
        
        # Normalisation
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        
        return df_original, df, df_encoded, X, y, label_encoders, scaler
    except Exception as e:
        st.error(f"❌ Erreur : Placez le fichier 'Données_Et_Dictionnaire_-.xlsx' dans le même dossier")
        st.stop()

# Fonction pour entraîner le modèle avec validation croisée et SMOTE
@st.cache_resource
def train_model_with_validation(X, y):
    # Split initial
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    
    # SMOTE pour équilibrer les classes (améliore la généralisation)
    smote = SMOTE(random_state=0)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Modèle optimisé
    model = RandomForestClassifier(
        bootstrap=False, 
        max_depth=8, 
        n_estimators=17, 
        random_state=0,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    # Validation croisée (K-Fold stratifié)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring='f1')
    
    # Entraînement sur les données équilibrées
    model.fit(X_train_balanced, y_train_balanced)
    
    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métriques
    metrics = {
        'accuracy_train': accuracy_score(y_train, y_pred_train),
        'accuracy_test': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test),
        'recall': recall_score(y_test, y_pred_test),
        'f1': f1_score(y_test, y_pred_test),
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred_test,
        'y_pred_proba': y_pred_proba,
        'overfitting': accuracy_score(y_train, y_pred_train) - accuracy_score(y_test, y_pred_test)
    }
    
    return model, metrics, X_train.columns.tolist()

# Chargement des données au démarrage
if st.session_state.df_original is None:
    with st.spinner("🔄 Chargement des données..."):
        df_original, df, df_encoded, X, y, label_encoders, scaler = load_and_prepare_data()
        st.session_state.df_original = df_original
        st.session_state.df = df
        st.session_state.df_encoded = df_encoded
        st.session_state.label_encoders = label_encoders
        
    with st.spinner("🤖 Entraînement du modèle avec validation croisée..."):
        model, metrics, feature_names = train_model_with_validation(X, y)
        st.session_state.model = model
        st.session_state.metrics = metrics
        st.session_state.feature_names = feature_names
        st.session_state.X = X
        st.session_state.y = y

# SIDEBAR
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: black; font-size: 28px;'>👥 Churn RH</h1>
            <p style='color: #2F2F2F;'>Prédiction de l'attrition</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Info données
    if st.session_state.df_original is not None:
        st.info(f"📊 {len(st.session_state.df_original)} employés")
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 📋 Navigation")
    
    if st.button("🎯 Objectif", use_container_width=True, type="primary" if st.session_state.page == "🎯 Objectif" else "secondary"):
        st.session_state.page = "🎯 Objectif"
        st.rerun()
    
    if st.button("📊 Dashboard", use_container_width=True, type="primary" if st.session_state.page == "📊 Dashboard" else "secondary"):
        st.session_state.page = "📊 Dashboard"
        st.rerun()
    
    if st.button("🔮 Prédictions", use_container_width=True, type="primary" if st.session_state.page == "🔮 Prédictions" else "secondary"):
        st.session_state.page = "🔮 Prédictions"
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #95a5a6; font-size: 12px; padding: 20px 0;'>
            <p>Fait par Junior Notam</p>
        </div>
    """, unsafe_allow_html=True)

# Titre
st.title(f"{st.session_state.page}")
st.markdown("---")

# PAGE: OBJECTIF
if st.session_state.page == "🎯 Objectif":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎓 Contexte
        Ce projet vise à **prédire l'attrition des employés** (churn) dans une organisation 
        en utilisant des techniques de Machine Learning avancées avec **validation croisée**.
        
        ### 🎯 Problématique
        L'attrition des employés représente un coût important pour les entreprises :
        - 💰 **Coûts de recrutement** élevés
        - 📉 **Perte de productivité** temporaire
        - 🧠 **Perte de connaissances** organisationnelles
        - 😟 **Impact sur le moral** des équipes
        
        ### ✨ Solution Proposée
        Notre système utilise un **modèle Random Forest** avec :
        - 🔄 **Validation croisée K-Fold** pour meilleure généralisation
        - ⚖️ **SMOTE** pour équilibrer les classes
        - 📊 **Métriques robustes** (Precision, Recall, F1-Score)
        
        ### 📈 Bénéfices
        - ⚡ **Détection précoce** des employés à risque
        - 🎯 **Actions ciblées** de rétention
        - 📊 **Insights** sur les facteurs de churn
        - 💡 **Optimisation** des stratégies RH
        """)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Métriques Clés</h3>
            <p>Notre modèle analyse</p>
            <h2>30+</h2>
            <p>variables RH</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.session_state.metrics:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Généralisation</h3>
                <p>Validation croisée</p>
                <h2>{st.session_state.metrics['cv_mean']:.1%}</h2>
                <p>±{st.session_state.metrics['cv_std']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Méthodologie
    st.subheader("🔬 Méthodologie")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #e3f2fd; border-radius: 10px;'>
            <h3>1️⃣</h3>
            <h4>Collecte</h4>
            <p>Données RH</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #f3e5f5; border-radius: 10px;'>
            <h3>2️⃣</h3>
            <h4>Préparation</h4>
            <p>Nettoyage</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #fff3e0; border-radius: 10px;'>
            <h3>3️⃣</h3>
            <h4>Équilibrage</h4>
            <p>SMOTE</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #e8f5e9; border-radius: 10px;'>
            <h3>4️⃣</h3>
            <h4>Validation</h4>
            <p>K-Fold CV</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #fce4ec; border-radius: 10px;'>
            <h3>5️⃣</h3>
            <h4>Déploiement</h4>
            <p>Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

# PAGE: DASHBOARD
elif st.session_state.page == "📊 Dashboard":
    # Sous-onglets
    tab1, tab2 = st.tabs(["📈 Dashboard Descriptif", "🔬 Dashboard Technique"])
    
    # TAB 1: DASHBOARD DESCRIPTIF
    with tab1:
        st.subheader("📊 Analyse Descriptive Interactive")
        
        # FILTRES
        st.markdown("### 🎛️ Filtres")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            departments = ['Tous'] + list(st.session_state.df['Department'].unique())
            selected_dept = st.selectbox("🏛️ Département", departments)
        
        with col2:
            genders = ['Tous'] + list(st.session_state.df['Gender'].unique())
            selected_gender = st.selectbox("⚧ Genre", genders)
        
        with col3:
            marital = ['Tous'] + list(st.session_state.df['MaritalStatus'].unique())
            selected_marital = st.selectbox("💑 Statut Marital", marital)
        
        with col4:
            overtime = ['Tous'] + list(st.session_state.df['OverTime'].unique())
            selected_overtime = st.selectbox("⏰ Heures Sup", overtime)
        
        # Appliquer les filtres
        df_filtered = st.session_state.df.copy()
        
        if selected_dept != 'Tous':
            df_filtered = df_filtered[df_filtered['Department'] == selected_dept]
        if selected_gender != 'Tous':
            df_filtered = df_filtered[df_filtered['Gender'] == selected_gender]
        if selected_marital != 'Tous':
            df_filtered = df_filtered[df_filtered['MaritalStatus'] == selected_marital]
        if selected_overtime != 'Tous':
            df_filtered = df_filtered[df_filtered['OverTime'] == selected_overtime]
        
        st.info(f"📊 {len(df_filtered)} employés après filtrage")
        
        st.markdown("---")
        
        # GRAPHIQUES DESCRIPTIFS
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎨 Distribution de l'Attrition")
            attrition_counts = df_filtered['Attrition'].value_counts()
            fig = px.pie(
                values=attrition_counts.values,
                names=['No', 'Yes'],
                color_discrete_sequence=['#4CAF50', '#FF5252'],
                hole=0.4,
                title=f"Taux d'attrition: {(attrition_counts.get('Yes', 0) / len(df_filtered) * 100):.1f}%"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🏛️ Attrition par Département")
            dept_attrition = df_filtered.groupby(['Department', 'Attrition']).size().reset_index(name='Count')
            fig = px.bar(
                dept_attrition,
                x='Department',
                y='Count',
                color='Attrition',
                barmode='group',
                color_discrete_map={'No': '#4CAF50', 'Yes': '#FF5252'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚧ Attrition par Genre")
            gender_attrition = df_filtered.groupby(['Gender', 'Attrition']).size().reset_index(name='Count')
            fig = px.bar(
                gender_attrition,
                x='Gender',
                y='Count',
                color='Attrition',
                barmode='group',
                color_discrete_map={'No': '#4CAF50', 'Yes': '#FF5252'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 💑 Attrition par Statut Marital")
            marital_attrition = df_filtered.groupby(['MaritalStatus', 'Attrition']).size().reset_index(name='Count')
            fig = px.bar(
                marital_attrition,
                x='MaritalStatus',
                y='Count',
                color='Attrition',
                barmode='group',
                color_discrete_map={'No': '#4CAF50', 'Yes': '#FF5252'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 Distribution de l'Âge")
            fig = px.histogram(
                df_filtered,
                x='Age',
                color='Attrition',
                nbins=20,
                barmode='overlay',
                color_discrete_map={'No': '#4CAF50', 'Yes': '#FF5252'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 💰 Distribution du Revenu")
            fig = px.histogram(
                df_filtered,
                x='MonthlyIncome',
                color='Attrition',
                nbins=20,
                barmode='overlay',
                color_discrete_map={'No': '#4CAF50', 'Yes': '#FF5252'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✈️ Attrition par Voyages Pro")
            travel_attrition = df_filtered.groupby(['BusinessTravel', 'Attrition']).size().reset_index(name='Count')
            fig = px.bar(
                travel_attrition,
                x='BusinessTravel',
                y='Count',
                color='Attrition',
                barmode='group',
                color_discrete_map={'No': '#4CAF50', 'Yes': '#FF5252'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### ⏰ Attrition par Heures Sup")
            overtime_attrition = df_filtered.groupby(['OverTime', 'Attrition']).size().reset_index(name='Count')
            fig = px.bar(
                overtime_attrition,
                x='OverTime',
                y='Count',
                color='Attrition',
                barmode='group',
                color_discrete_map={'No': '#4CAF50', 'Yes': '#FF5252'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Box plots
        st.markdown("#### 📦 Comparaison des Variables Numériques")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                df_filtered,
                x='Attrition',
                y='YearsAtCompany',
                color='Attrition',
                color_discrete_map={'No': '#4CAF50', 'Yes': '#FF5252'},
                title="Années dans l'entreprise"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df_filtered,
                x='Attrition',
                y='TotalWorkingYears',
                color='Attrition',
                color_discrete_map={'No': '#4CAF50', 'Yes': '#FF5252'},
                title="Années d'expérience totale"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: DASHBOARD TECHNIQUE
    with tab2:
        st.subheader("🔬 Analyse Technique du Modèle")
        
        # Métriques de généralisation
        st.markdown("### 📊 Métriques de Performance et Généralisation")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🎯 Accuracy Test", f"{st.session_state.metrics['accuracy_test']:.2%}")
        with col2:
            st.metric("🎪 Précision", f"{st.session_state.metrics['precision']:.2%}")
        with col3:
            st.metric("📈 Recall", f"{st.session_state.metrics['recall']:.2%}")
        with col4:
            st.metric("⚡ F1-Score", f"{st.session_state.metrics['f1']:.2%}")
        with col5:
            overfit = st.session_state.metrics['overfitting']
            st.metric("🔄 Overfitting", f"{overfit:.2%}", 
                     delta="Bon" if overfit < 0.05 else "Attention",
                     delta_color="normal" if overfit < 0.05 else "inverse")
        
        st.markdown("---")
        
        # Validation croisée
        st.markdown("### 🔄 Validation Croisée (5-Fold)")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("📊 F1-Score Moyen", f"{st.session_state.metrics['cv_mean']:.2%}")
            st.metric("📉 Écart-type", f"{st.session_state.metrics['cv_std']:.2%}")
            st.info("✅ La validation croisée montre que le modèle **généralise bien** sur différents sous-ensembles de données.")
        
        with col2:
            cv_scores = st.session_state.metrics['cv_scores']
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f'Fold {i+1}' for i in range(len(cv_scores))],
                y=cv_scores,
                marker_color='#667eea',
                text=[f'{score:.2%}' for score in cv_scores],
                textposition='outside'
            ))
            fig.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="red", 
                         annotation_text=f"Moyenne: {cv_scores.mean():.2%}")
            fig.update_layout(
                title="Scores F1 par Fold",
                yaxis_title="F1-Score",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Graphiques techniques
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔥 Matrice de Confusion")
            cm = confusion_matrix(st.session_state.metrics['y_test'], st.session_state.metrics['y_pred'])
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Prédit: Resté', 'Prédit: Parti'],
                y=['Réel: Resté', 'Réel: Parti'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20}
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Courbe ROC")
            fpr, tpr, _ = roc_curve(st.session_state.metrics['y_test'], st.session_state.metrics['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC (AUC = {roc_auc:.3f})',
                line=dict(color='#4CAF50', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Aléatoire',
                line=dict(color='gray', dash='dash')
            ))
            fig.update_layout(
                title=f'Courbe ROC (AUC = {roc_auc:.3f})',
                xaxis_title='Taux de Faux Positifs',
                yaxis_title='Taux de Vrais Positifs',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        st.markdown("#### 🌟 Importance des Variables (Top 15)")
        feature_importance = pd.DataFrame({
            'Feature': st.session_state.feature_names,
            'Importance': st.session_state.model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Viridis',
            title="Les 15 variables les plus importantes pour la prédiction"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Matrice de corrélation des top features
        st.markdown("#### 🔗 Matrice de Corrélation (Top 10 Features)")
        top_features = feature_importance.head(10)['Feature'].tolist()
        
        # Récupérer les indices des colonnes
        feature_indices = [st.session_state.feature_names.index(f) for f in top_features]
        X_top = st.session_state.X.iloc[:, feature_indices]
        X_top.columns = top_features
        
        corr_matrix = X_top.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig.update_layout(
            title="Corrélation entre les variables importantes",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

# PAGE: PRÉDICTIONS
elif st.session_state.page == "🔮 Prédictions":
    st.markdown("### Entrez les informations de l'employé")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("👤 Âge", 18, 65, 30)
        monthly_income = st.number_input("💰 Revenu Mensuel ($)", 1000, 20000, 5000, 100)
        distance_from_home = st.slider("🏠 Distance du domicile (km)", 1, 30, 10)
        num_companies_worked = st.slider("🏢 Nb d'entreprises précédentes", 0, 10, 2)
        
    with col2:
        years_at_company = st.slider("📅 Années dans l'entreprise", 0, 40, 5)
        total_working_years = st.slider("💼 Années d'expérience totale", 0, 40, 10)
        years_since_last_promotion = st.slider("🎖️ Années depuis dernière promo", 0, 15, 2)
        years_with_curr_manager = st.slider("👔 Années avec manager actuel", 0, 17, 3)
        
    with col3:
        business_travel = st.selectbox("✈️ Voyages professionnels", 
                                      ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
        department = st.selectbox("🏛️ Département",
                                 ['Sales', 'Research & Development', 'Human Resources'])
        gender = st.selectbox("⚧ Genre", ['Male', 'Female'])
        marital_status = st.selectbox("💑 Statut marital",
                                     ['Single', 'Married', 'Divorced'])
        overtime = st.selectbox("⏰ Heures supplémentaires", ['No', 'Yes'])
    
    # Bouton de prédiction
    if st.button("🚀 Prédire le Risque de Churn", type="primary", use_container_width=True):
        # Créer le DataFrame avec toutes les features nécessaires
        input_data = pd.DataFrame(columns=st.session_state.feature_names)
        
        # Remplir avec des valeurs par défaut
        for col in st.session_state.feature_names:
            input_data[col] = [0]
        
        # Mapper les valeurs encodées
        business_travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
        department_map = {'Human Resources': 0, 'Research & Development': 1, 'Sales': 2}
        gender_map = {'Female': 0, 'Male': 1}
        marital_map = {'Divorced': 0, 'Married': 1, 'Single': 2}
        overtime_map = {'No': 0, 'Yes': 1}
        
        # Assigner les valeurs
        if 'Age' in input_data.columns:
            input_data['Age'] = age
        if 'MonthlyIncome' in input_data.columns:
            input_data['MonthlyIncome'] = monthly_income
        if 'DistanceFromHome' in input_data.columns:
            input_data['DistanceFromHome'] = distance_from_home
        if 'NumCompaniesWorked' in input_data.columns:
            input_data['NumCompaniesWorked'] = num_companies_worked
        if 'YearsAtCompany' in input_data.columns:
            input_data['YearsAtCompany'] = years_at_company
        if 'TotalWorkingYears' in input_data.columns:
            input_data['TotalWorkingYears'] = total_working_years
        if 'YearsSinceLastPromotion' in input_data.columns:
            input_data['YearsSinceLastPromotion'] = years_since_last_promotion
        if 'YearsWithCurrManager' in input_data.columns:
            input_data['YearsWithCurrManager'] = years_with_curr_manager
        if 'BusinessTravel' in input_data.columns:
            input_data['BusinessTravel'] = business_travel_map.get(business_travel, 0)
        if 'Department' in input_data.columns:
            input_data['Department'] = department_map.get(department, 0)
        if 'Gender' in input_data.columns:
            input_data['Gender'] = gender_map.get(gender, 0)
        if 'MaritalStatus' in input_data.columns:
            input_data['MaritalStatus'] = marital_map.get(marital_status, 0)
        if 'OverTime' in input_data.columns:
            input_data['OverTime'] = overtime_map.get(overtime, 0)
        
        # Prédiction
        prediction = st.session_state.model.predict(input_data)[0]
        proba = st.session_state.model.predict_proba(input_data)[0]
        
        # Affichage des résultats
        st.markdown("---")
        st.markdown("### 📊 Résultats de la Prédiction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box">
                    ⚠️ RISQUE ÉLEVÉ DE CHURN<br>
                    Probabilité: {proba[1]:.1%}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    ✅ RISQUE FAIBLE DE CHURN<br>
                    Probabilité: {proba[0]:.1%}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Jauge de probabilité
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=proba[1] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risque de Churn (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if proba[1] > 0.5 else "darkgreen"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommandations
        st.markdown("### 💡 Recommandations")
        
        if prediction == 1:
            st.warning("""
            **Actions recommandées :**
            - 🎯 Organiser un entretien individuel rapidement
            - 💰 Évaluer les possibilités d'augmentation ou de bonus
            - 📈 Proposer un plan de développement de carrière
            - 🎓 Offrir des opportunités de formation
            - 🏆 Reconnaissance et valorisation du travail accompli
            """)
        else:
            st.success("""
            **Bonnes pratiques à maintenir :**
            - ✅ Continuer le suivi régulier
            - 🤝 Maintenir un bon environnement de travail
            - 📊 Évaluations de performance périodiques
            - 🌱 Opportunités de croissance continue
            """)