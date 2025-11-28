# app.py
import streamlit as st
import numpy as np
from PIL import Image
from process_tshirt import process_tshirt
from threshold_config import get_config


def pil_to_numpy(pil_image):
    """Convert PIL Image to numpy array in BGR format for OpenCV compatibility."""
    rgb_array = np.array(pil_image)
    if len(rgb_array.shape) == 2:  # Grayscale
        return rgb_array
    return rgb_array[:, :, ::-1]  # RGB to BGR


def numpy_to_pil(numpy_array):
    """Convert numpy array from BGR to RGB PIL Image."""
    if len(numpy_array.shape) == 2:  # Grayscale
        return Image.fromarray(numpy_array)
    return Image.fromarray(numpy_array[:, :, ::-1])  # BGR to RGB


def main():
    st.set_page_config(
        page_title="Print Defect Detection",
        page_icon="👕",
        layout="wide"
    )
    
    st.title("👕 Print Defect Detection")
    st.markdown("---")
    
    # Create two columns for upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Golden Sample")
        golden_file = st.file_uploader(
            "Upload Golden Sample Image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="golden"
        )
        if golden_file:
            st.image(golden_file, caption="Golden Sample", width="stretch")
    
    with col2:
        st.subheader("📸 Test Sample")
        test_file = st.file_uploader(
            "Upload Test Sample Image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="test"
        )
        if test_file:
            st.image(test_file, caption="Test Sample", width="stretch")
    
    st.markdown("---")
    
    # Detect button
    detect_button = st.button("🔍 Detect Defects", type="primary", use_container_width=True)
    
    if detect_button:
        if not golden_file or not test_file:
            st.error("⚠️ Please upload both Golden Sample and Test Sample images!")
        else:
            with st.spinner("🔄 Processing images and detecting defects..."):
                try:
                    # Load images directly into memory
                    golden_image = Image.open(golden_file)
                    test_image = Image.open(test_file)
                    
                    # Convert to numpy arrays
                    golden_array = pil_to_numpy(golden_image)
                    test_array = pil_to_numpy(test_image)
                    
                    # Load config and run detection
                    cfg = get_config()
                    result = process_tshirt(golden_array, test_array, cfg)
                    
                    st.success("✅ Detection completed successfully!")
                    
                    # Display results
                    st.markdown("---")
                    st.header("📊 Detection Results")
                    
                    # Display statistics in columns
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric("Defect Status", "DEFECT" if result["is_defect"] else "PASS")
                    
                    with metric_col2:
                        st.metric("Mean ΔE", f"{result['mean_diff']:.2f}")
                    
                    with metric_col3:
                        st.metric("Max ΔE", f"{result['max_diff']:.2f}")
                    
                    with metric_col4:
                        st.metric("Defect Area %", f"{result['filtered_percent']:.2f}%")
                    
                    st.markdown("---")
                    
                    # Display images in three columns
                    st.subheader("🖼️ Visual Results")
                    img_col1, img_col2, img_col3 = st.columns(3)
                    
                    with img_col1:
                        st.markdown("**Delta-E Normalized**")
                        st.image(result["delta_e_normalized"], width="stretch", clamp=True)
                    
                    with img_col2:
                        st.markdown("**Final Heatmap**")
                        st.image(numpy_to_pil(result["heatmap"]), width="stretch")
                    
                    with img_col3:
                        st.markdown("**Defect Overlay**")
                        st.image(numpy_to_pil(result["overlay"]), width="stretch")
                    
                    st.markdown("---")
                    
                    # Show additional images in expander
                    with st.expander("🔍 View Additional Analysis Images"):
                        add_col1, add_col2 = st.columns(2)
                        
                        with add_col1:
                            st.markdown("**Aligned Test Image**")
                            st.image(numpy_to_pil(result["aligned"]), width="stretch")
                            
                            st.markdown("**Delta-E Map**")
                            st.image(result["delta_e_map"], width="stretch", clamp=True)
                            
                            st.markdown("**Defects (Unfiltered)**")
                            st.image(result["defect_mask_unfiltered"], width="stretch", clamp=True)
                        
                        with add_col2:
                            st.markdown("**Defects (Filtered)**")
                            st.image(result["defect_mask_filtered"], width="stretch", clamp=True)
                            
                            st.markdown("**Heatmap Visualization**")
                            st.image(numpy_to_pil(result["heatmap"]), width="stretch")
                
                except Exception as e:
                    st.error(f"❌ Error during detection: {str(e)}")
                    st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Print Defect Detection System | Powered by Computer Vision & Delta-E Analysis</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
