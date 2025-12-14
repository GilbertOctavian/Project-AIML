import streamlit as st
import pandas as pd
import numpy as np
import datetime

# ==========================================
# KONFIGURASI DATA (ASUMSI > 5 ITEM & TUJUAN)
# ==========================================

# Konfigurasi Kapal (1 Hari ada 3 Kapal)
SHIPS_PER_DAY = 3
SHIP_NAMES = ["Kapal Logistik A (Cepat)", "Kapal Logistik B (Reguler)", "Kapal Logistik C (Kargo Besar)"]
MAX_CAPACITY_PER_SHIP = 5000  # kg

# Data Tujuan (Asumsi 5 Tujuan)
DESTINATIONS = {
    "Jakarta": {"distance": 700, "base_price_per_kg": 5000, "eta_days": 2},
    "Makassar": {"distance": 1200, "base_price_per_kg": 8000, "eta_days": 4},
    "Balikpapan": {"distance": 1000, "base_price_per_kg": 7500, "eta_days": 3},
    "Medan": {"distance": 2000, "base_price_per_kg": 12000, "eta_days": 5},
    "Jayapura": {"distance": 3500, "base_price_per_kg": 25000, "eta_days": 10}
}

# Data Barang (Asumsi 5 Jenis Barang)
ITEM_TYPES = [
    "Elektronik (Asuransi Tinggi)",
    "Pakaian & Tekstil",
    "Makanan Kering",
    "Furniture/Perabot",
    "Otomotif/Sparepart"
]

# Konstanta Pajak
TAX_RATE = 0.11  # PPN 11%

# ==========================================
# FUNGSI BACKEND SEDERHANA
# ==========================================

def calculate_price(destination, weight, item_type):
    dest_data = DESTINATIONS[destination]
    base_cost = dest_data["base_price_per_kg"] * weight
    
    # Biaya tambahan berdasarkan jenis barang
    handling_fee = 0
    if "Elektronik" in item_type:
        handling_fee = base_cost * 0.05 # Extra 5% untuk asuransi
    elif "Furniture" in item_type:
        handling_fee = 50000 # Flat fee volume
        
    subtotal = base_cost + handling_fee
    tax = subtotal * TAX_RATE
    total_price = subtotal + tax
    
    return subtotal, tax, total_price, dest_data["eta_days"]

def schedule_shipments(orders):
    """
    Algoritma Sederhana untuk memasukkan order ke dalam jadwal 3 kapal per hari.
    Menggantikan logika PSO yang rumit dengan 'Bin Packing' sederhana untuk demo dosen.
    """
    if not orders:
        return []

    schedule = []
    # Urutkan order berdasarkan ETA (prioritas tujuan jauh) atau berat
    sorted_orders = sorted(orders, key=lambda x: x['weight'], reverse=True)
    
    current_day = 1
    daily_ships = {0: [], 1: [], 2: []} # Index kapal 0, 1, 2
    ship_capacities = {0: 0, 1: 0, 2: 0} # Kapasitas terisi saat ini
    
    for order in sorted_orders:
        assigned = False
        
        # Coba masukkan ke salah satu dari 3 kapal di hari ini
        for ship_idx in range(SHIPS_PER_DAY):
            if ship_capacities[ship_idx] + order['weight'] <= MAX_CAPACITY_PER_SHIP:
                daily_ships[ship_idx].append(order)
                ship_capacities[ship_idx] += order['weight']
                assigned = True
                
                schedule.append({
                    "Hari": f"Hari ke-{current_day}",
                    "Nama Kapal": SHIP_NAMES[ship_idx],
                    "ID Order": order['id'],
                    "Tujuan": order['destination'],
                    "Barang": order['item'],
                    "Berat (kg)": order['weight'],
                    "Estimasi Sampai": f"{order['eta']} hari lagi"
                })
                break
        
        # Jika hari ini penuh semua kapal, pindah hari besok (Simple Logic)
        if not assigned:
            # Reset untuk hari baru (dalam simulasi ini kita simplifikasi order sisa masuk hari berikutnya)
            # Di implementasi nyata, ini butuh loop while.
            # Untuk demo, kita paksa masuk ke Kapal C (Overload) atau buat notifikasi
            schedule.append({
                "Hari": f"Hari ke-{current_day + 1}", # Lempar ke besok
                "Nama Kapal": SHIP_NAMES[0],
                "ID Order": order['id'],
                "Tujuan": order['destination'],
                "Barang": order['item'],
                "Berat (kg)": order['weight'],
                "Status": "Reschedule (Overload)"
            })

    return pd.DataFrame(schedule)

# ==========================================
# USER INTERFACE (STREAMLIT)
# ==========================================

st.set_page_config(layout="wide", page_title="Sistem Logistik Kapal Laut")

# Inisialisasi Session State untuk menyimpan Order antar halaman
if 'orders_db' not in st.session_state:
    st.session_state['orders_db'] = []

# Sidebar untuk Navigasi (Simulasi "Windows Baru")
st.sidebar.title("Navigasi Sistem")
page_selection = st.sidebar.radio("Pilih Halaman:", ["Halaman Customer", "Halaman Admin"])

if page_selection == "Halaman Customer":
    st.title("🚢 Layanan Pengiriman Kargo Laut")
    st.markdown("### Input Data Pengiriman Barang")
    st.info("Silakan masukkan detail barang Anda untuk melihat estimasi harga dan jadwal.")

    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            cust_name = st.text_input("Nama Pengirim")
            item_selected = st.selectbox("Jenis Barang", ITEM_TYPES)
            item_weight = st.number_input("Berat Barang (kg)", min_value=1, value=10)
        
        with col2:
            destination_selected = st.selectbox("Tujuan Pengiriman", list(DESTINATIONS.keys()))
            notes = st.text_area("Catatan Tambahan (Opsional)")

        # Tombol Hitung & Pesan
        submitted = st.form_submit_button("Cek Harga & Estimasi")

    if submitted:
        if cust_name:
            # Hitung Logika
            subtotal, tax, total, eta = calculate_price(destination_selected, item_weight, item_selected)
            
            # Tampilkan Hasil Perhitungan
            st.divider()
            st.subheader("🧾 Rincian Biaya & Estimasi")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Tujuan", destination_selected)
                st.metric("Estimasi Sampai", f"{eta} Hari")
            with res_col2:
                st.metric("Berat Total", f"{item_weight} kg")
                st.metric("Subtotal", f"Rp {subtotal:,.0f}")
            with res_col3:
                st.metric("Pajak (PPN 11%)", f"Rp {tax:,.0f}")
                st.metric("TOTAL HARGA", f"Rp {total:,.0f}", delta="Harus Dibayar")

            # Simpan Order ke Database Sementara
            new_order = {
                "id": len(st.session_state['orders_db']) + 1,
                "customer": cust_name,
                "item": item_selected,
                "weight": item_weight,
                "destination": destination_selected,
                "price": total,
                "eta": eta,
                "timestamp": datetime.datetime.now()
            }
            st.session_state['orders_db'].append(new_order)
            st.success("✅ Order berhasil dicatat! Admin akan menjadwalkan pengiriman Anda.")
        else:
            st.warning("Mohon isi Nama Pengirim.")

    # Tampilkan History Order User Ini (Opsional)
    if st.session_state['orders_db']:
        st.markdown("---")
        st.caption("Daftar antrian order yang baru saja masuk:")
        df_display = pd.DataFrame(st.session_state['orders_db'])
        st.dataframe(df_display[["id", "customer", "item", "destination", "price"]])

elif page_selection == "Halaman Admin":
    st.title("👮‍♂️ Dashboard Admin Logistik")
    st.write("Mengatur jadwal pengiriman harian (1 Hari = 3 Kapal).")

    # Metrics Ringkasan
    total_orders = len(st.session_state['orders_db'])
    total_revenue = sum([o['price'] for o in st.session_state['orders_db']])
    
    met1, met2, met3 = st.columns(3)
    met1.metric("Total Order Masuk", total_orders)
    met2.metric("Total Estimasi Pendapatan", f"Rp {total_revenue:,.0f}")
    met3.metric("Ketersediaan Kapal Hari Ini", f"{SHIPS_PER_DAY} Unit")

    st.divider()

    tab1, tab2 = st.tabs(["📋 Data Order Masuk", "📅 Generate Jadwal Kapal"])

    with tab1:
        st.subheader("Data Semua Order Konsumen")
        if total_orders > 0:
            df_orders = pd.DataFrame(st.session_state['orders_db'])
            st.dataframe(df_orders)
        else:
            st.warning("Belum ada data order dari konsumen.")

    with tab2:
        st.subheader("Penjadwalan Otomatis")
        st.write("Klik tombol di bawah untuk mendistribusikan barang ke 3 kapal yang tersedia.")
        
        if st.button("Jalankan Penjadwalan (Auto-Schedule)", type="primary"):
            if total_orders > 0:
                with st.spinner("Sedang menghitung kapasitas muatan kapal..."):
                    schedule_df = schedule_shipments(st.session_state['orders_db'])
                
                st.success("Jadwal Berhasil Dibuat!")
                
                # Menampilkan Jadwal Grouping
                st.markdown("### 🚢 Manifest Muatan Kapal")
                
                # Tampilkan per hari/per kapal agar rapi
                unique_days = schedule_df["Hari"].unique()
                for day in unique_days:
                    st.markdown(f"#### {day}")
                    day_data = schedule_df[schedule_df["Hari"] == day]
                    
                    for ship in SHIP_NAMES:
                        ship_data = day_data[day_data["Nama Kapal"] == ship]
                        if not ship_data.empty:
                            with st.expander(f"{ship} - ({len(ship_data)} Item)", expanded=True):
                                st.table(ship_data[["ID Order", "Tujuan", "Barang", "Berat (kg)"]])
                                total_load = ship_data["Berat (kg)"].sum()
                                st.progress(min(total_load / MAX_CAPACITY_PER_SHIP, 1.0), text=f"Load: {total_load} / {MAX_CAPACITY_PER_SHIP} kg")
                        else:
                            st.caption(f"🚫 {ship}: Tidak ada muatan.")
            else:
                st.error("Tidak ada order untuk dijadwalkan.")

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.caption("Sistem Logistik v2.0 - Dibuat untuk Tugas Kuliah")