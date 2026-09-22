import numpy as np

def Tmdh(para):
    """
    :param para: List of MDH parameters [a, alpha, d, theta].
    :return: 4x4 transformation matrix.
    """
    a, alpha, d, theta = para
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, a],
        [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
        [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
        [0, 0, 0, 1]
    ])

def jnttocart(q1, q2, q3, q4, q5, q6, q7,isleft):
    # Constants dual arm
    awr = 0.0905
    dbs = 0.2047
    dse = 0.3
    dew = 0.26

    # Modified Denavit-Hartenberg parameters
    ParamMDH = [
        [0, 0, dbs, q1],
        [0, -np.pi/2, 0, q2 - np.pi/2],
        [0, np.pi/2, -dse, q3 + np.pi/2],
        [0, np.pi/2, 0, q4],
        [0, -np.pi/2, -dew, q5 - np.pi/2],
        [0, -np.pi/2, 0, q6 + np.pi/2],
        [awr, np.pi/2, 0, q7]
    ]

    # Compute transformation matrices
    if isleft:
        Tf0 = np.array([
        [0,  0, 1, 0],
        [0,  1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1]])
    else:
        Tf0 = np.array([
        [0,  0, -1, 0],
        [0,  -1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1]])

    T7e = np.array([
    [0,  0, -1, 0.0495],
    [0,  1, 0, 0],
    [1, 0, 0, -0.0101],
    [0, 0, 0, 1]])

    T01 = Tmdh(ParamMDH[0])
    T12 = Tmdh(ParamMDH[1])
    T23 = Tmdh(ParamMDH[2])
    T34 = Tmdh(ParamMDH[3])
    T45 = Tmdh(ParamMDH[4])
    T56 = Tmdh(ParamMDH[5])
    T67 = Tmdh(ParamMDH[6])

    # Compute overall transformation matrix T07
    Tfe = Tf0 @ T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T67 @ T7e
  
    epsilon = 1e-12

    pitch = np.arctan2(-Tfe[2, 0], np.sqrt(Tfe[0, 0]**2 + Tfe[1, 0]**2))

    if abs(pitch) > (np.pi / 2 - epsilon):
        yaw = np.arctan2(-Tfe[0, 1], Tfe[1, 1])
        roll = 0.0
    else:
        roll = np.arctan2(Tfe[2, 1], Tfe[2, 2])
        yaw = np.arctan2(Tfe[1, 0], Tfe[0, 0])

    # return Tfe
    # return np.array([Tfe[0, 3],Tfe[1, 3],Tfe[2, 3], roll*180/np.pi, pitch*180/np.pi, yaw*180/np.pi])
    return np.array([Tfe[0, 3],Tfe[1, 3],Tfe[2, 3], roll, pitch, yaw])

# Example usage
# q1, q2, q3, q4, q5, q6, q7 = -np.pi/4, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, 0 # Example joint angles in radians
# redun = jnttocart(q1, q2, q3, q4, q5, q6, q7,True)
# print(redun)
# redun = jnttocart(q1, q2, q3, q4, q5, q6, q7,False)
# print(redun)