import plotly.graph_objects as go
import plotly.io as pio


def plot_point_cloud(xyz, name='default', save=False, path=None, voxels=None, boxes=None,
                     semantics=None, pred_centers=None, box_centers=None, default_zoom=1.25,
                     position_coefs=[0,0,0], marker_size=3, img_size=None, contour_width=3,
                     scale=1):
    """Plots point clouds with voxel centers and bounding boxes."""
    pio.renderers.default = "browser"
    figures = []  # figures to display

    # Draw point cloud.
    pcolors = '#000000' if (semantics is None) else semantics
    fig = go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode='markers',
        marker=dict(size=marker_size,
                    color=pcolors,
                    line=dict(
                        color='rgba(255, 255, 255, 1)',
                        width=contour_width
                    ))
    )
    figures.append(fig)

    if voxels is not None:
        # Add voxel centers.
        fig = go.Scatter3d(
            x=voxels[:, 0],
            y=voxels[:, 1],
            z=voxels[:, 2],
            mode='markers',
            marker=dict(size=2.5, color='#40E3EC')
        )
        figures.append(fig)

    if pred_centers is not None:
        # Add centers predicted draw box centers.
        pred_centers = go.Scatter3d(
            x=pred_centers[:, 0],
            y=pred_centers[:, 1],
            z=pred_centers[:, 2],
            mode='markers',
            marker=dict(color='#eb4034', size=2)
        )
        figures.append(pred_centers)

    if boxes is not None:
        box_figures, box_edges = [], []
        for i, box in enumerate(boxes):
            # Draw box edges pairwise between corners.
            p1, p2, p3, p4, p1_, p2_, p3_, p4_ = box
            lines = [[p1, p2], [p1, p4], [p1_, p2_], [p1_, p4_],
                     [p3, p2], [p3, p4], [p3_, p2_], [p3_, p4_],
                     [p1, p1_], [p2, p2_], [p3, p3_], [p4, p4_]]
            for line in lines:
                tmp = go.Scatter3d(
                    x=[line[0][0], line[1][0]],
                    y=[line[0][1], line[1][1]],
                    z=[line[0][2], line[1][2]],
                    mode='lines',
                    line=dict(width=2.5, color='#59fcff')
                )
                box_edges.append(tmp)

        if box_centers is not None:
            # Draw box centers.
            box_cent = go.Scatter3d(
                x=box_centers[:, 0],
                y=box_centers[:, 1],
                z=box_centers[:, 2],
                mode='markers',
                marker=dict(color='#2D2D2D', size=3)
            )
            figures.append(box_cent)

        # Create figure.
        fig = go.Figure(data=figures + box_edges)
    else:
        fig = go.Figure(data=figures)

        # Set default zoom to given range.
        camera = dict(
            eye=dict(x=default_zoom+position_coefs[0],
                     y=default_zoom+position_coefs[1],
                     z=default_zoom+position_coefs[2]))
        fig.update_layout(scene_camera=camera, title=name)

        if save:
            pio.write_image(fig, path, width=img_size, height=img_size, scale=scale)

    fig.show()
