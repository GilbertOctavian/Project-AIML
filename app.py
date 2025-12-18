import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt

# ==========================================
# CONFIG & CONSTANTS
# ==========================================
st.set_page_config(layout="wide", page_title="Port Logistics AI", initial_sidebar_state="expanded")

# CSS Custom: Styling Kotak KPI agar berwarna-warni
st.markdown("""
<style>
    .stProgress > div > div > div > div { background-color: #1f77b4; }
    
    /* Base Box Style */
    .kpi-box {
        padding: 15px; 
        border-radius: 8px; 
        margin-bottom: 10px;
    }
    
    /* Hijau (Uang/Revenue) */
    .box-green {
        background-color: #d1e7dd; 
        border-left: 5px solid #198754;
        color: #0f5132;
    }
    
    /* Biru (Berat/Logistik) */
    .box-blue {
        background-color: #cff4fc; 
        border-left: 5px solid #0dcaf0;
        color: #055160;
    }
    
    /* Kuning (Antrian/Order) */
    .box-yellow {
        background-color: #fff3cd; 
        border-left: 5px solid #ffc107;
        color: #664d03;
    }

    .kpi-label { font-size: 14px; font-weight: bold; opacity: 0.8; }
    .kpi-value { font-size: 26px; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# Simulasi Armada: Feeder Vessel
SHIPS = ["KM. Meratus Jaya", "KM. Tanto Line", "KM. SPIL Nusantara"]
CAPACITY_PER_SHIP = 500000 # 500.000 kg (500 Ton)
DAILY_CAPACITY = len(SHIPS) * CAPACITY_PER_SHIP
PLAN_DAYS = 3 

# Rute & Base Rate
ROUTES = {
    "Tanjung Priok (JKT)":  {"dist": 0,    "rate": 2500, "days": 1},
    "Soekarno-Hatta (MKS)": {"dist": 1400, "rate": 4800, "days": 4},
    "Semayang (BPN)":       {"dist": 1200, "rate": 4200, "days": 3},
    "Belawan (MDN)":        {"dist": 1800, "rate": 5800, "days": 5},
    "Jayapura (DJJ)":       {"dist": 3700, "rate": 9800, "days": 12}
}

# Tipe Container
CONTAINER_TYPES = {
    "LCL / Pallet (1 Ton)": 1000,
    "Dry Container 20ft (20 Ton)": 20000,
    "Dry Container 40ft (25 Ton)": 25000,
    "Reefer / Frozen (25 Ton)": 25000 # Priority
}

GOODS_TYPE = ["General Cargo", "Electronics", "Textile", "FMCG", "Automotive Parts"]
PPN = 0.11

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_quote(route, weight, goods, container_type):
    route_info = ROUTES[route]
    base_price = route_info["rate"] * weight
    
    # Surcharge logic
    surcharge = 0
    if "Reefer" in container_type:
        surcharge = 3500000 
    elif "Electronics" in goods:
        surcharge = base_price * 0.08 
        
    subtotal = base_price + surcharge
    tax = subtotal * PPN
    grand_total = subtotal + tax
    
    return subtotal, tax, grand_total, route_info["days"]

def mock_data_gen(n=20):
    clients = ["PT. Indofood", "Mayora Group", "Unilever Indo", "Astra Honda", "Semen Gresik", "Wings Food", "Erajaya", "Gudang Garam"]
    data = []
    for _ in range(n):
        c_name = random.choices(list(CONTAINER_TYPES.keys()), weights=[30, 40, 15, 15], k=1)[0]
        qty = random.randint(1, 2)
        weight = CONTAINER_TYPES[c_name] * qty
        dest = random.choice(list(ROUTES.keys()))
        item = random.choice(GOODS_TYPE)
        
        sub, tax, total, eta = get_quote(dest, weight, item, c_name)
        
        data.append({
            "id": 0, 
            "client": random.choice(clients),
            "item": item, "type": c_name,
            "qty": qty, "weight": weight, "dest": dest,
            "total": total, "eta": eta
        })
    return data

# ==========================================
# PSO ALGORITHM
# ==========================================

def pso_scheduler(orders, particles=20, iterations=30):
    if not orders: return [], []
    dim = len(orders)
    w, c1, c2 = 0.7, 1.4, 1.4 
    
    X = [np.random.rand(dim) for _ in range(particles)]
    V = [np.random.rand(dim) * 0.1 for _ in range(particles)]
    
    pbest_X = X[:]
    pbest_score = [float('inf')] * particles
    gbest_X = X[0]
    gbest_score = float('inf')
    loss_history = []

    def calc_cost(position_vector):
        queue = []
        for i, val in enumerate(position_vector):
            prio_score = val
            if "Reefer" in orders[i]['type']:
                prio_score += 10.0 
            queue.append((prio_score, orders[i]))
            
        queue.sort(key=lambda x: x[0], reverse=True)
        ship_loads = {d: [0]*len(SHIPS) for d in range(1, PLAN_DAYS + 2)}
        total_penalty = 0
        
        for _, order in queue:
            assigned = False
            for day in range(1, PLAN_DAYS + 1):
                current_day_loads = ship_loads[day]
                best_ship_idx = current_day_loads.index(min(current_day_loads))
                
                if current_day_loads[best_ship_idx] + order['weight'] <= CAPACITY_PER_SHIP:
                    ship_loads[day][best_ship_idx] += order['weight']
                    delay = day - 1
                    cost = (delay * order['weight']) / 1000 
                    
                    if "Reefer" in order['type'] and delay > 0:
                        cost += 5000000 
                    
                    total_penalty += cost
                    assigned = True
                    break
            
            if not assigned:
                total_penalty += (order['weight']/1000) * 10000
        
        all_loads = [load for d in ship_loads for load in ship_loads[d]]
        total_penalty += np.std(all_loads) / 100
        return total_penalty

    for i in range(iterations):
        for p in range(particles):
            r1, r2 = random.random(), random.random()
            V[p] = w*V[p] + c1*r1*(pbest_X[p] - X[p]) + c2*r2*(gbest_X - X[p])
            X[p] = np.clip(X[p] + V[p], 0, 1)
            
            score = calc_cost(X[p])
            if score < pbest_score[p]:
                pbest_score[p] = score
                pbest_X[p] = X[p]
                if score < gbest_score:
                    gbest_score = score
                    gbest_X = X[p]
        loss_history.append(gbest_score)
    
    final_schedule = []
    final_queue = []
    for i, val in enumerate(gbest_X):
        prio = val + (10.0 if "Reefer" in orders[i]['type'] else 0)
        final_queue.append((prio, orders[i]))
    final_queue.sort(key=lambda x: x[0], reverse=True)
    
    loads = {d: [0]*len(SHIPS) for d in range(1, PLAN_DAYS + 2)}
    for _, order in final_queue:
        assigned = False
        for day in range(1, PLAN_DAYS + 1):
            s_idx = loads[day].index(min(loads[day]))
            if loads[day][s_idx] + order['weight'] <= CAPACITY_PER_SHIP:
                loads[day][s_idx] += order['weight']
                status = "On Schedule"
                if "Reefer" in order['type']:
                    status = "🔥 PRIORITY (Reefer)" if day == 1 else "⚠️ SPOILED (Late)"
                elif day > 1:
                    status = f"Reschedule H+{day-1}"
                
                final_schedule.append({**order, "Day": day, "Ship": SHIPS[s_idx], "Status": status})
                assigned = True
                break
        if not assigned:
            final_schedule.append({**order, "Day": 99, "Ship": "BACKLOG", "Status": "REJECTED"})
            
    return pd.DataFrame(final_schedule), loss_history

# ==========================================
# UI / FRONTEND
# ==========================================
if 'db' not in st.session_state: st.session_state['db'] = []

with st.sidebar:
    st.header("🚢 Logistic System")
    page = st.radio("Navigation", ["Customer Booking", "Ops Dashboard"])
    st.divider()
    if page == "Ops Dashboard":
        st.subheader("Dev Tools")
        if st.button("⚡ Generate 20 Mock Orders"):
            new_data = mock_data_gen(20)
            for d in new_data:
                d['id'] = len(st.session_state['db']) + 1
                st.session_state['db'].append(d)
            st.success("20 Orders Injected!")
            time.sleep(0.5)
            st.rerun()
        if st.button("🗑️ Flush Database"):
            st.session_state['db'] = []
            st.rerun()

# PAGE 1: CUSTOMER
if page == "Customer Booking":
    st.title("📦 Cargo Booking Portal")
    st.write("Input detail muatan untuk mendapatkan *Quotation*.")
    
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            client = st.text_input("Nama Perusahaan")
            dest = st.selectbox("Pelabuhan Tujuan", list(ROUTES.keys()))
            item = st.selectbox("Jenis Barang", GOODS_TYPE)
        with c2:
            ctype = st.selectbox("Tipe Kontainer", list(CONTAINER_TYPES.keys()))
            qty = st.number_input("Jumlah Unit", 1, 50, 1)
            
            w_per_unit = CONTAINER_TYPES[ctype]
            total_w = w_per_unit * qty
            sub, tax, grand_tot, eta = get_quote(dest, total_w, item, ctype)
            
            st.info(f"Total Weight: **{total_w/1000:,.1f} Ton**")
            if "Reefer" in ctype:
                st.warning("❄️ Reefer Container selected (High Priority Handling)")

    st.divider()
    
    # PRICING SECTION (GREEN BOX)
    k1, k2, k3 = st.columns(3)
    k1.metric("Subtotal", f"Rp {sub:,.0f}")
    k2.metric("PPN (11%)", f"Rp {tax:,.0f}")
    
    with k3:
        st.markdown(f"""
        <div class="kpi-box box-green">
            <span class="kpi-label">TOTAL NETT</span><br>
            <span class="kpi-value">Rp {grand_tot:,.0f}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("") 
    
    if st.button("Submit Booking", type="primary", use_container_width=True):
        if not client:
            st.error("Nama Perusahaan wajib diisi!")
        else:
            payload = {
                "id": len(st.session_state['db']) + 1,
                "client": client, "item": item, "type": ctype,
                "qty": qty, "weight": total_w, "dest": dest,
                "total": grand_tot, "eta": eta
            }
            st.session_state['db'].append(payload)
            st.success("Booking Confirmed! DO (Delivery Order) has been issued.")
            
            with st.expander("📄 View Invoice", expanded=True):
                st.markdown(f"""
                **INVOICE #{payload['id']}** Client: {client} | Route: {dest}  
                Cargo: {qty}x {ctype} ({item})  
                **Total Paid: Rp {grand_tot:,.0f}**
                """)

# PAGE 2: ADMIN DASHBOARD
elif page == "Ops Dashboard":
    st.title("⚓ Operations Control Tower")
    df = st.session_state['db']
    
    # KPI CARDS (COLORED BOXES)
    total_ton = sum(x['weight'] for x in df) / 1000 if df else 0
    est_rev = sum(x['total'] for x in df) if df else 0
    pending_count = len(df)
    
    col1, col2, col3 = st.columns(3)
    
    # BOX 1: PENDING ORDER (YELLOW)
    with col1:
        st.markdown(f"""
        <div class="kpi-box box-yellow">
            <span class="kpi-label">PENDING DO</span><br>
            <span class="kpi-value">{pending_count} Order</span>
        </div>
        """, unsafe_allow_html=True)

    # BOX 2: TOTAL TONNAGE (BLUE)
    with col2:
        st.markdown(f"""
        <div class="kpi-box box-blue">
            <span class="kpi-label">TOTAL TONNAGE</span><br>
            <span class="kpi-value">{total_ton:,.1f} Ton</span>
        </div>
        """, unsafe_allow_html=True)

    # BOX 3: EST REVENUE (GREEN)
    with col3:
        st.markdown(f"""
        <div class="kpi-box box-green">
            <span class="kpi-label">EST. REVENUE</span><br>
            <span class="kpi-value">Rp {est_rev/1000000:,.1f} Jt</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    if not df:
        st.info("No orders in queue. Use 'Dev Tools' in sidebar to generate mock data.")
    else:
        lc, rc = st.columns([1, 2])
        with lc:
            st.subheader("🤖 AI Scheduler")
            st.caption("Algorithm: Particle Swarm Optimization (PSO)")
            st.write("Mengoptimalkan Load Balancing kapal dan memprioritaskan Reefer Container.")
            
            if st.button("Run Optimization", type="primary", use_container_width=True):
                with st.spinner("Calculating optimal stowage plan..."):
                    time.sleep(1) 
                    res_df, history = pso_scheduler(df)
                    st.session_state['res'] = res_df
                    st.session_state['hist'] = history
                st.success("Optimization Complete.")
            
            if 'hist' in st.session_state:
                st.markdown("### Cost Convergence")
                fig, ax = plt.subplots(figsize=(4,3))
                ax.plot(st.session_state['hist'], label='Penalty Score', color='#ff4b4b')
                ax.set_xlabel('Iterations')
                ax.set_ylabel('Cost')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

        with rc:
            st.subheader("📅 Stowage Plan (Jadwal Muat)")
            if 'res' in st.session_state:
                res = st.session_state['res']
                tabs = st.tabs(["DAY 1 (Priority)", "DAY 2", "DAY 3", "BACKLOG"])
                
                for i, tab in enumerate(tabs):
                    day = i + 1
                    with tab:
                        if day < 4: daily_data = res[res['Day'] == day]
                        else: daily_data = res[res['Day'] == 99]
                        
                        if daily_data.empty: st.write("No operations scheduled.")
                        else:
                            cols = st.columns(len(SHIPS)) if day < 4 else [st.container()]
                            ship_names = SHIPS if day < 4 else ["BACKLOG"]
                            if day == 4: st.error(f"Failed to load: {len(daily_data)} items")

                            for idx, s_name in enumerate(ship_names):
                                with (cols[idx] if day < 4 else st.container()):
                                    s_data = daily_data[daily_data['Ship'] == s_name]
                                    load = s_data['weight'].sum()
                                    st.markdown(f"**{s_name}**")
                                    if day < 4:
                                        pct = load / CAPACITY_PER_SHIP
                                        st.progress(min(pct, 1.0))
                                        st.caption(f"{load/1000:,.0f} / 500 Ton ({pct*100:.1f}%)")
                                    
                                    if not s_data.empty:
                                        display_df = s_data[['dest', 'type', 'total', 'Status']].copy()
                                        display_df['total'] = display_df['total'].apply(lambda x: f"{x/1e6:.1f}M")
                                        st.dataframe(display_df, hide_index=True, use_container_width=True)
                                    else:
                                        st.markdown("*Idle*")
            else:
                st.warning("Waiting for optimization trigger...")