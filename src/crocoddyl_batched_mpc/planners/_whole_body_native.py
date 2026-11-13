"""Optional native types for the standing W CPU reference.

Imported only when constructing the planner, never by the legacy Go2 path.
"""

import crocoddyl as c
import numpy as np


class TorqueMemoryState(c.StateAbstract):
    """Product manifold (multibody state, previous torque), dimensions 67/66.

    The final 18 Euclidean entries are memory, not physical coordinates. All
    physical integration is delegated to StateMultibody (49/48).
    """

    def __init__(self, physical, nu):
        super().__init__(physical.nx + nu, physical.ndx + nu)
        self.physical, self.memory_size = physical, nu

    def zero(self):
        return np.r_[self.physical.zero(), np.zeros(self.memory_size)]

    def rand(self):
        return np.r_[self.physical.rand(), np.random.uniform(-1, 1, self.memory_size)]

    def diff(self, x0, x1):
        n = self.physical.nx
        return np.r_[self.physical.diff(x0[:n], x1[:n]), x1[n:] - x0[n:]]

    def integrate(self, x, dx):
        n, d = self.physical.nx, self.physical.ndx
        return np.r_[self.physical.integrate(x[:n], dx[:d]), x[n:] + dx[d:]]

    def _extend(self, matrices, signs):
        d = self.physical.ndx
        result = []
        for matrix, sign in zip(matrices, signs):
            full = np.eye(self.ndx) * sign
            full[:d, :d] = matrix
            result.append(full)
        return result

    def Jdiff(self, x0, x1, firstsecond=c.Jcomponent.both):
        n = self.physical.nx
        signs = (
            [-1, 1]
            if firstsecond == c.Jcomponent.both
            else [-1 if firstsecond == c.Jcomponent.first else 1]
        )
        return self._extend(self.physical.Jdiff(x0[:n], x1[:n], firstsecond), signs)

    def Jintegrate(self, x, dx, firstsecond=c.Jcomponent.both):
        matrices = self.physical.Jintegrate(
            x[: self.physical.nx], dx[: self.physical.ndx], firstsecond
        )
        return self._extend(matrices, [1] * len(matrices))

    def JintegrateTransport(self, x, dx, Jin, firstsecond):
        return self.Jintegrate(x, dx, firstsecond)[0] @ Jin


class TorqueMemoryAction(c.ActionModelAbstract):
    """Exact adjacent-control cost using z[k+1]=u[k].

    Running cost adds dt*w/2*||u[k]-z[k]||^2. At k=0 z is the explicitly
    supplied previous torque (or the verified static seed). Terminal cost has
    no extra torque or memory penalty. Bounds act on absolute torque u.
    """

    def __init__(self, state, physical_action, weight, dt):
        super().__init__(state, physical_action.nu, 0)
        self.physical_action = physical_action
        self.slew_scale = weight * dt
        self.u_lb = physical_action.u_lb.copy()
        self.u_ub = physical_action.u_ub.copy()

    def createData(self):
        data = c.ActionDataAbstract(self)
        data.physical = self.physical_action.createData()
        return data

    def calc(self, data, x, u=None):
        n = self.state.physical.nx
        if u is None:
            self.physical_action.calc(data.physical, x[:n])
            data.xnext[:] = x
            data.cost = data.physical.cost
        else:
            self.physical_action.calc(data.physical, x[:n], u)
            delta = u - x[n:]
            data.xnext[:] = np.r_[data.physical.xnext, u]
            data.cost = data.physical.cost + 0.5 * self.slew_scale * delta.dot(delta)

    def calcDiff(self, data, x, u=None):
        n, d = self.state.physical.nx, self.state.physical.ndx
        if u is None:
            self.physical_action.calcDiff(data.physical, x[:n])
        else:
            self.physical_action.calcDiff(data.physical, x[:n], u)
        for name in ("Fx", "Fu", "Lx", "Lu", "Lxx", "Lxu", "Luu"):
            getattr(data, name)[:] = 0
        data.Lx[:d] = data.physical.Lx
        data.Lxx[:d, :d] = data.physical.Lxx
        if u is None:
            return
        data.Fx[:d, :d] = data.physical.Fx
        data.Fu[:d] = data.physical.Fu
        data.Fu[d:] = np.eye(self.nu)
        data.Lu[:] = data.physical.Lu
        data.Lxu[:d] = data.physical.Lxu
        data.Luu[:] = data.physical.Luu
        delta = u - x[n:]
        scale = self.slew_scale
        data.Lx[d:] = -scale * delta
        data.Lu[:] += scale * delta
        data.Lxx[d:, d:] = scale * np.eye(self.nu)
        data.Lxu[d:] = -scale * np.eye(self.nu)
        data.Luu[:] += scale * np.eye(self.nu)


class WorldForceConeResidual(c.ResidualModelAbstract):
    """Inscribed diamond |fx|+|fy| <= mu*fz with normal-force bounds.

    Requires ContactModel3D LOCAL_WORLD_ALIGNED and force derivatives enabled.
    ContactData3D.f and df_* then use world-aligned contact coordinates.
    """

    def __init__(self, state, name, nu, mu):
        super().__init__(state, 6, nu, True, True, True)
        self.name = name
        self.matrix = np.array(
            [[1, 1, -mu], [1, -1, -mu], [-1, 1, -mu], [-1, -1, -mu], [0, 0, -1], [0, 0, 1]],
            dtype=float,
        )

    def createData(self, shared):
        data = c.ResidualDataAbstract(self, shared)
        data.contact = shared.contacts.contacts[self.name]
        return data

    def calc(self, data, x, u=None):
        data.r[:] = self.matrix @ data.contact.f.linear

    def calcDiff(self, data, x, u=None):
        data.Rx[:] = self.matrix @ data.contact.df_dx[:3]
        data.Ru[:] = self.matrix @ data.contact.df_du[:3]
