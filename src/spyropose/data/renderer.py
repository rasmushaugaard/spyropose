import moderngl
import numpy as np
import trimesh


def orthographic_matrix(left, right, bottom, top, near, far):
    return np.array(
        (
            (2 / (right - left), 0, 0, -(right + left) / (right - left)),
            (0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)),
            (0, 0, -2 / (far - near), -(far + near) / (far - near)),
            (0, 0, 0, 1),
        )
    )


def projection_matrix(K, w, h, near=10.0, far=10000.0):  # 1 cm to 10 m
    # transform from cv2 camera coordinates to opengl (flipping sign of y and z)
    view = np.eye(4)
    view[1:3] *= -1

    # see http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    persp = np.zeros((4, 4))
    persp[:2, :3] = K[:2, :3]
    persp[2, 2:] = near + far, near * far
    persp[3, 2] = -1
    # transform the camera matrix from cv2 to opengl as well (flipping sign of y and z)
    persp[:2, 1:3] *= -1

    # The origin of the image is in the *center* of the top left pixel.
    # The orthographic matrix should map the whole image *area* into the opengl NDC, therefore the -.5 below:
    orth = orthographic_matrix(-0.5, w - 0.5, -0.5, h - 0.5, near, far)
    return orth @ persp @ view


class SimpleRenderer:
    def __init__(
        self,
        mesh: trimesh.Trimesh,
        near: float,
        far: float,
        w: int,
        h: int = None,
        device_idx=0,
        components=4,
        dtype="f4",
    ):
        self.mesh = mesh
        self.components = components
        self.dtype = dtype
        if h is None:
            h = w
        self.h, self.w = h, w
        self.ctx = moderngl.create_context(
            standalone=True, backend="egl", device_index=device_idx
        )
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.fbo = self.ctx.simple_framebuffer(
            (w, h), components=components, dtype=dtype
        )
        self.near, self.far = near, far

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                uniform mat3 R;
                uniform mat4 mvp;
                uniform mat4 mv;
                in vec3 in_vert;
                in vec3 in_normal;
                out vec3 normal;
                out vec3 vert;
                void main() {
                    vert = (mv * vec4(in_vert, 1.0)).xyz;
                    normal = R * in_normal;
                    gl_Position = mvp * vec4(in_vert, 1.0);
                }
                """,
            fragment_shader="""
                #version 330
                in vec3 normal;
                out vec4 fragColor;
                in vec3 vert;
                void main() {
                    float c = 0.1 + 0.85 * max(0, -dot(normal, normalize(vert)));
                    fragColor = vec4(c, c, c, 1.0);
                }
                """,
        )

        self.vao = self.ctx.simple_vertex_array(
            self.prog,
            self.ctx.buffer(
                np.concatenate(
                    (
                        mesh.vertices[mesh.faces],  # (n, 3 verts, 3 xyz)
                        np.repeat(mesh.face_normals[:, None], 3, axis=1),
                        # mesh.vertex_normals[mesh.faces],
                    ),
                    axis=-1,
                ).astype(np.float32)
            ),
            "in_vert",
            "in_normal",
        )

    def read(self):
        return

    def render(self, K, R, t):
        mv = np.concatenate(
            (
                np.concatenate((R, t), axis=1),
                [[0, 0, 0, 1]],
            )
        )
        mvp = projection_matrix(K, self.w, self.h, self.near, self.far) @ mv
        self.prog["R"].value = tuple(R.T.astype("f4").reshape(-1))
        self.prog["mvp"].value = tuple(mvp.T.astype("f4").reshape(-1))
        self.prog["mv"].value = tuple(mv.T.astype("f4").reshape(-1))

        self.fbo.use()
        self.ctx.clear()
        self.vao.render(mode=moderngl.TRIANGLES)

        return np.frombuffer(
            self.fbo.read(components=self.components, dtype=self.dtype),
            dict(f4="f4", f2="f2", f1="u1")[self.dtype],
        ).reshape((self.h, self.w, self.components))
