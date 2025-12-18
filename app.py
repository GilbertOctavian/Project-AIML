import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import random
import matplotlib.pyplot as plt

# ==========================================
# KONFIGURASI DATA
# ==========================================

# Konfigurasi Kapal (1 Hari ada 3 Kapal)
SHIPS_PER_DAY = 3
SHIP_NAMES = ["Kapal Logistik A (Cepat)", "Kapal Logistik B (Reguler)", "Kapal Logistik C (Kargo Besar)"]
MAX_CAPACITY_PER_SHIP = 5000  # kg

# Data Tujuan
DESTINATIONS = {
    "Jakarta": {"distance": 700, "base_price_per_kg": 5000, "eta_days": 2},
    "Makassar": {"distance": 1200, "base_price_per_kg": 8000, "eta_days": 4},
    "Balikpapan": {"distance": 1000, "base_price_per_kg": 7500, "eta_days": 3},
    "Medan": {"distance": 2000, "base_price_per_kg": 12000, "eta_days": 5},
    "Jayapura": {"distance": 3500, "base_price_per_kg": 25000, "eta_days": 10}
}

# Data Barang
ITEM_TYPES = [
    "Elektronik (Asuransi Tinggi)",
    "Pakaian & Tekstil",
    "Makanan Kering",
    "Furniture/Perabot",
    "Otomotif/Sparepart"
]

TAX_RATE = 0.11

# ==========================================
# FUNGSI BACKEND
# ==========================================

def calculate_price(destination, weight, item_type):
    dest_data = DESTINATIONS[destination]
    base_cost = dest_data["base_price_per_kg"] * weight
    
    handling_fee = 0
    if "Elektronik" in item_type:
        handling_fee = base_cost * 0.05
    elif "Furniture" in item_type:
        handling_fee = 50000
        
    subtotal = base_cost + handling_fee
    tax = subtotal * TAX_RATE
    total_price = subtotal + tax
    
    return subtotal, tax, total_price, dest_data["eta_days"]

# --- GREEDY ALGORITHM (SIMPLE & FAST) ---
def schedule_shipments_greedy(orders):
    if not orders:
        return pd.DataFrame()

    schedule = []
    # Urutkan dari berat terbesar (Priority)
    sorted_orders = sorted(orders, key=lambda x: x['weight'], reverse=True)
    
    current_day = 1
    ship_capacities = {0: 0, 1: 0, 2: 0}
    
    for order in sorted_orders:
        # LOGIKA LEAST LOADED (Cari kapal yang paling kosong)
        best_ship_idx = -1
        min_load = float('inf')
        
        for i in range(SHIPS_PER_DAY):
            # Cek muat gak?
            if ship_capacities[i] + order['weight'] <= MAX_CAPACITY_PER_SHIP:
                # Cari yang load-nya paling kecil
                if ship_capacities[i] < min_load:
                    min_load = ship_capacities[i]
                    best_ship_idx = i
        
        if best_ship_idx != -1:
            ship_capacities[best_ship_idx] += order['weight']
            schedule.append({
                "Hari": f"Hari ke-{current_day}",
                "Nama Kapal": SHIP_NAMES[best_ship_idx],
                "ID Order": order['id'],
                "Tujuan": order['destination'],
                "Barang": order['item'],
                "Berat Total (kg)": order['weight'],
                "Estimasi Sampai": f"{order['eta']} hari lagi",
                "Status": "Terjadwal (OK)"
            })
        else:
            schedule.append({
                "Hari": f"Hari ke-{current_day + 1}",
                "Nama Kapal": "Waiting List",
                "ID Order": order['id'],
                "Tujuan": order['destination'],
                "Barang": order['item'],
                "Berat Total (kg)": order['weight'],
                "Status": "Reschedule (Overload)",
                "Estimasi Sampai": "-"
            })

    return pd.DataFrame(schedule)

# --- ALGORITMA PSO (PARTICLE SWARM OPTIMIZATION) ---
def run_pso_optimization(orders, num_particles=20, iterations=30):
    # Parameter PSO Standard
    w = 0.8    # Inertia (Seberapa besar partikel mempertahankan arah lama)
    c1 = 1.5   # Cognitive (Belajar dari pengalaman sendiri)
    c2 = 1.5   # Social (Belajar dari teman terbaik/GBest)
    
    dim = len(orders) # Setiap dimensi merepresentasikan prioritas 1 order
    
    # --- FUNGSI FITNESS (PENENTU KUALITAS JADWAL) ---
    # Semakin KECIL nilai fitness, semakin BAGUS jadwalnya.
    def calculate_fitness(priorities):
        # 1. Mapping Priority dari Partikel ke Order
        orders_with_prio = []
        for i, order in enumerate(orders):
            o = order.copy()
            o['prio'] = priorities[i]
            orders_with_prio.append(o)
        
        # 2. Urutkan Order berdasarkan Priority yang dibuat PSO
        # Ini adalah inti "Otak" AI-nya: Mencari urutan masuk terbaik
        sorted_by_pso = sorted(orders_with_prio, key=lambda x: x['prio'], reverse=True)
        
        # 3. Simulasi Masukin Barang ke Kapal
        ship_caps = {0: 0, 1: 0, 2: 0} # Kapasitas terpakai
        reschedule_weight = 0
        temp_schedule = []
        
        for order in sorted_by_pso:
            # Strategi: Masukkan ke kapal yang BEBANNYA PALING RINGAN (Load Balancing)
            best_ship_idx = -1
            min_load = float('inf')
            
            for i in range(SHIPS_PER_DAY):
                # Syarat: Harus muat
                if ship_caps[i] + order['weight'] <= MAX_CAPACITY_PER_SHIP:
                    # Cari yang paling kosong
                    if ship_caps[i] < min_load:
                        min_load = ship_caps[i]
                        best_ship_idx = i
            
            if best_ship_idx != -1:
                ship_caps[best_ship_idx] += order['weight']
                temp_schedule.append({
                    "Hari": f"Hari ke-1",
                    "Nama Kapal": SHIP_NAMES[best_ship_idx],
                    "ID Order": order['id'],
                    "Tujuan": order['destination'],
                    "Barang": order['item'],
                    "Berat Total (kg)": order['weight'],
                    "Estimasi Sampai": f"{order['eta']} hari lagi",
                    "Status": "Terjadwal (OK)"
                })
            else:
                # Gagal muat -> Kena Penalty
                reschedule_weight += order['weight']
                temp_schedule.append({
                    "Hari": f"Hari ke-2",
                    "Nama Kapal": "Waiting List",
                    "ID Order": order['id'],
                    "Tujuan": order['destination'],
                    "Barang": order['item'],
                    "Berat Total (kg)": order['weight'],
                    "Status": "Reschedule (Overload)",
                    "Estimasi Sampai": "-"
                })
        
        # --- MENGHITUNG NILAI FITNESS (SCORE ERROR) ---
        # 1. Penalty Utama: Berat yang gagal angkut (Harus diminimalkan!)
        # 2. Penalty Tambahan: Ketimpangan beban antar kapal (Standard Deviasi)
        #    Agar grafik tetap bergerak turun walau semua barang sudah muat.
        
        loads = list(ship_caps.values()) # Contoh: [1000, 5000, 200]
        load_imbalance = np.std(loads)   # Mengukur seberapa timpang bebannya
        
        # Rumus Fitness: (Berat Gagal * 1000) + (Ketimpangan Beban)
        # Kita kali 1000 supaya "Barang Masuk" jadi prioritas nomor 1.
        fitness_score = (reschedule_weight * 1000) + load_imbalance
        
        return fitness_score, reschedule_weight, temp_schedule

    # --- INISIALISASI PSO ---
    # Partikel disebar random
    particles = [np.random.rand(dim) for _ in range(num_particles)]
    velocities = [np.random.rand(dim) * 0.1 for _ in range(num_particles)]
    
    pbest_pos = particles[:]
    pbest_scores = [calculate_fitness(p)[0] for p in particles]
    
    # Cari GBest awal (Juara sementara)
    gbest_score = min(pbest_scores)
    gbest_idx = pbest_scores.index(gbest_score)
    gbest_pos = pbest_pos[gbest_idx]
    
    fitness_history = []
    
    # --- LOOP ITERASI (PROSES BELAJAR AI) ---
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for t in range(iterations):
        for i in range(num_particles):
            # Update Kecepatan & Posisi Partikel
            r1, r2 = random.random(), random.random()
            
            # Rumus Fisika PSO: v_baru = w*v_lama + c1*belajar_sendiri + c2*belajar_dari_gbest
            velocities[i] = w * velocities[i] + \
                            c1 * r1 * (pbest_pos[i] - particles[i]) + \
                            c2 * r2 * (gbest_pos - particles[i])
            
            particles[i] = particles[i] + velocities[i]
            particles[i] = np.clip(particles[i], 0, 1) # Tahan agar tidak keluar batas
            
            # Cek nilai fitness baru
            current_fitness, _, _ = calculate_fitness(particles[i])
            
            # Update Personal Best (PBest)
            if current_fitness < pbest_scores[i]:
                pbest_scores[i] = current_fitness
                pbest_pos[i] = particles[i]
                
                # Update Global Best (GBest)
                if current_fitness < gbest_score:
                    gbest_score = current_fitness
                    gbest_pos = particles[i]
        
        # Simpan sejarah fitness untuk grafik
        fitness_history.append(gbest_score)
        
        # Ambil data real weight untuk display status
        _, gbest_real_weight, _ = calculate_fitness(gbest_pos)
        
        status_text.text(f"Iterasi {t+1}/{iterations} - Error Score: {gbest_score:.2f}")
        progress_bar.progress((t + 1) / iterations)
    
    # Ambil hasil akhir dari posisi terbaik
    _, _, final_schedule = calculate_fitness(gbest_pos)
    return pd.DataFrame(final_schedule), fitness_history

# ==========================================
# USER INTERFACE
# ==========================================

st.set_page_config(layout="wide", page_title="Sistem Logistik Kapal Laut")

if 'orders_db' not in st.session_state:
    st.session_state['orders_db'] = []

# Sidebar
st.sidebar.title("Navigasi Sistem")
page_selection = st.sidebar.radio("Pilih Halaman:", ["Halaman Input Order", "Dashboard Admin"])

# --- HALAMAN 1: INPUT ORDER ---
if page_selection == "Halaman Input Order":
    st.title("🚢 Input Order Pengiriman")
    st.info("💡 TIPS: Masukkan minimal 5-10 order dengan berat bervariasi agar grafik PSO terlihat bergerak!")

    st.sidebar.header("Panel Input Batch")
    num_orders_input = st.sidebar.slider("Jumlah Pelanggan/Order:", 1, 15, 5) # Default 5 biar langsung banyak
    
    with st.form("batch_order_form"):
        batch_orders = []
        for i in range(num_orders_input):
            with st.expander(f"Order Pelanggan #{i+1}", expanded=(i==0)):
                c_name = st.text_input(f"Nama Pengirim #{i+1}", key=f"name_{i}", value=f"Pelanggan {i+1}")
                c_item = st.selectbox(f"Jenis Barang #{i+1}", ITEM_TYPES, key=f"item_{i}")
                c_dest = st.selectbox(f"Tujuan #{i+1}", list(DESTINATIONS.keys()), key=f"dest_{i}")
                
                col_qty, col_weight = st.columns(2)
                with col_qty:
                    c_qty = st.number_input(f"Jumlah Barang (Qty) #{i+1}", min_value=1, value=1, step=1, key=f"q_{i}")
                with col_weight:
                    # Random value default biar user ga capek ngisi satu2
                    def_val = random.randint(10, 100) * 10
                    c_weight_per_item = st.number_input(f"Berat per Item (kg) #{i+1}", min_value=1, value=def_val, step=10, key=f"w_{i}")
                
                batch_orders.append({
                    "name": c_name, "item": c_item, "dest": c_dest, 
                    "qty": c_qty, "weight_per_item": c_weight_per_item
                })
        
        if st.form_submit_button("🚀 Proses Semua Order"):
            with st.spinner("Sedang menghitung biaya..."):
                time.sleep(0.5)
                count_success = 0
                for order in batch_orders:
                    if order["name"]:
                        total_weight = order["qty"] * order["weight_per_item"]
                        subtotal, tax, total, eta = calculate_price(order["dest"], total_weight, order["item"])
                        
                        new_entry = {
                            "id": len(st.session_state['orders_db']) + 1,
                            "customer": order["name"],
                            "item": order["item"],
                            "qty": order["qty"],
                            "unit_weight": order["weight_per_item"],
                            "weight": total_weight,
                            "destination": order["dest"],
                            "price": total,
                            "eta": eta,
                            "timestamp": datetime.datetime.now()
                        }
                        st.session_state['orders_db'].append(new_entry)
                        count_success += 1
                
                if count_success > 0:
                    st.success(f"Berhasil menambahkan {count_success} order baru!")
                else:
                    st.warning("Mohon isi nama pengirim.")

    if st.session_state['orders_db']:
        st.divider()
        st.subheader("📋 Daftar Antrian Order")
        df_display = pd.DataFrame(st.session_state['orders_db'])
        if not df_display.empty:
            cols = ["id", "customer", "destination", "item", "qty", "unit_weight", "weight", "price"]
            df_show = df_display[cols]
            df_show.columns = ["ID", "Pengirim", "Tujuan", "Barang", "Qty", "Berat/Item", "Total Berat", "Total Biaya"]
            st.dataframe(df_show)

# --- HALAMAN 2: DASHBOARD ADMIN ---
elif page_selection == "Dashboard Admin":
    st.title("👮‍♂️ Dashboard Admin & Optimasi AI")
    
    total_orders = len(st.session_state['orders_db'])
    total_revenue = sum([o['price'] for o in st.session_state['orders_db']])
    
    st.markdown("### Rangkuman Finansial")
    c1, c2 = st.columns(2)
    c1.metric("Total Order Masuk", f"{total_orders} Unit")
    c2.metric("Total Estimasi Revenue", f"Rp {total_revenue:,.0f}")

    st.divider()
    
    tab_grafik, tab_jadwal = st.tabs(["📈 Grafik Statistik", "🤖 Optimasi Jadwal (PSO)"])

    with tab_grafik:
        st.subheader("Analisis Data Pengiriman")
        if total_orders > 0:
            df_chart = pd.DataFrame(st.session_state['orders_db'])
            cc1, cc2 = st.columns(2)
            with cc1:
                st.write("**Sebaran Tujuan**")
                st.bar_chart(df_chart['destination'].value_counts())
            with cc2:
                st.write("**Trend Pendapatan**")
                st.line_chart(df_chart['price'])
        else:
            st.warning("Belum ada data.")

    with tab_jadwal:
        st.subheader("Penjadwalan Pengiriman")
        st.write("Pilih metode untuk menyusun muatan kapal:")
        
        col_algo, col_act = st.columns([1, 2])
        with col_algo:
            algo_choice = st.radio("Metode Algoritma:", ["Greedy (Cepat)", "PSO (Machine Learning)"])
        
        with col_act:
            st.info("Greedy: Cepat & simple. \nPSO: Mencari solusi terbaik lewat iterasi (lebih lama tapi cerdas).")
            run_btn = st.button("Jalankan Optimasi", type="primary")

        if run_btn:
            if total_orders > 0:
                schedule_df = pd.DataFrame()
                
                if algo_choice == "Greedy (Cepat)":
                    with st.spinner("Menjalankan Algoritma Greedy..."):
                        time.sleep(1)
                        schedule_df = schedule_shipments_greedy(st.session_state['orders_db'])
                        st.success("Jadwal Greedy Selesai!")
                        
                elif algo_choice == "PSO (Machine Learning)":
                    st.write("---")
                    st.markdown("##### ⚙️ Proses Training PSO")
                    with st.spinner("Sedang melatih partikel..."):
                        # JALANKAN PSO
                        schedule_df, history = run_pso_optimization(st.session_state['orders_db'])
                    
                    st.success("Optimasi PSO Selesai!")
                    
                    st.markdown("##### 📉 Grafik Konvergensi Fitness PSO")
                    st.caption("Grafik yang **MENURUN** artinya AI sedang belajar mengurangi error (menyeimbangkan muatan).")
                    
                    # Plotting Grafik
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(history, marker='o', linestyle='-', color='#007acc', markersize=4)
                    ax.set_title("Perkembangan Nilai Fitness (Mencari Error Terkecil)")
                    ax.set_xlabel("Iterasi (Waktu Belajar)")
                    ax.set_ylabel("Nilai Error (Penalty + Ketimpangan)")
                    ax.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig)
                
                # --- HASIL JADWAL ---
                if not schedule_df.empty:
                    # Penanganan Error Kolom Kosong
                    if "Status" not in schedule_df.columns: schedule_df["Status"] = "Terjadwal (OK)"
                    if "Estimasi Sampai" not in schedule_df.columns: schedule_df["Estimasi Sampai"] = "-"

                    st.write("---")
                    st.subheader("📅 Hasil Penjadwalan Final")
                    
                    # Cek Reschedule
                    rescheduled_items = schedule_df[schedule_df["Status"].astype(str).str.contains("Reschedule", na=False)]
                    if len(rescheduled_items) == 0:
                        st.success("✅ Semua barang berhasil dijadwalkan!")
                    else:
                        st.warning(f"⚠️ Ada {len(rescheduled_items)} barang harus di-reschedule.")

                    # Tampilkan Data per Hari
                    for day in sorted(schedule_df["Hari"].unique()):
                        st.markdown(f"#### 🗓️ {day}")
                        day_data = schedule_df[schedule_df["Hari"] == day]
                        
                        cols_to_show = ["Nama Kapal", "Tujuan", "Barang", "Berat Total (kg)", "Estimasi Sampai"]
                        if day_data["Status"].astype(str).str.contains("Reschedule").any():
                            cols_to_show.append("Status")
                        
                        st.dataframe(day_data[cols_to_show])
                else:
                    st.error("Gagal membuat jadwal.")
            else:
                st.error("Data order kosong. Silakan input data terlebih dahulu.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Sistem Logistik v5.0 (Final Fix)")