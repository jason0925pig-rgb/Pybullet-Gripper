"""GraspLab GUI demonstration using mock grasp records (NOT PyBullet)."""
import random
import sys
from pathlib import Path

from PyQt6 import uic
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTableWidgetItem, QWidget

BASE_DIR = Path(__file__).resolve().parent


class MainWindow(QMainWindow):
    def __init__(self, login_window, username):
        super().__init__()
        self.login_window = login_window
        self.username = username
        self.records = []  # Temporary sample data; no database or persistence yet.
        self.pending_records = []
        self.completed = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_mock_simulation)
        uic.loadUi(str(BASE_DIR / 'main_window.ui'), self)
        self.welcomeLabel.setText(f"Hi, {self.username.capitalize()}!")
        self.pages = {
            'dashboard': self.dashboardPage,
            'simulation': self.simulationPage,
            'dataset': self.datasetPage,
            'ml': self.mlTrainingPage,
            'reports': self.reportsPage,
        }
        for button, name in (
            (self.dashboardButton, 'dashboard'),
            (self.simulationButton, 'simulation'),
            (self.datasetButton, 'dataset'),
            (self.mlButton, 'ml'),
            (self.reportsButton, 'reports'),
            (self.newSimulationButton, 'simulation'),
            (self.quickTrainButton, 'ml'),
        ):
            button.clicked.connect(lambda _checked=False, key=name: self.show_page(key))
        self.logoutButton.clicked.connect(self.logout)
        self.startSimulationButton.clicked.connect(self.start_mock_simulation)
        self.refreshDatasetButton.clicked.connect(self.refresh_dataset)
        self.resultFilterComboBox.currentIndexChanged.connect(self.refresh_dataset)
        self.graspTableWidget.setEditTriggers(self.graspTableWidget.EditTrigger.NoEditTriggers)
        self.show_page('dashboard')
        self.refresh_dataset()

    def show_page(self, name):
        self.stackedWidget.setCurrentWidget(self.pages[name])
        if name == 'dataset':
            self.refresh_dataset()

    def start_mock_simulation(self):
        if self.timer.isActive():
            return
        count = self.graspSpinBox.value()
        gripper = self.gripperComboBox.currentText()
        obj = self.objectComboBox.currentText()
        rng = random.Random()
        self.pending_records = [
            {
                'x': round(rng.uniform(-0.2, 0.2), 3),
                'y': round(rng.uniform(-0.2, 0.2), 3),
                'z': round(rng.uniform(0.15, 0.5), 3),
                'roll': round(rng.uniform(-1.5, 1.5), 3),
                'pitch': round(rng.uniform(-1.5, 1.5), 3),
                'yaw': round(rng.uniform(-3.14, 3.14), 3),
                'result': rng.randint(0, 1),
                'gripper': gripper,
                'object': obj,
            }
            for _ in range(count)
        ]
        self.completed = 0
        self.startSimulationButton.setEnabled(False)
        self.simulationProgressBar.setValue(0)
        self.totalGraspsLabel.setText('Total Grasps: 0')
        self.successfulGraspsLabel.setText('Successful: 0')
        self.failedGraspsLabel.setText('Failed: 0')
        self.statusBar().showMessage('MOCK simulation running — not connected to PyBullet')
        self.timer.start(60)

    def advance_mock_simulation(self):
        total = len(self.pending_records)
        # Make the mock run take about 1 second regardless of requested attempts.
        self.completed = min(total, self.completed + max(1, (total + 19) // 20))
        self.simulationProgressBar.setValue(round(self.completed * 100 / total))
        current = self.pending_records[:self.completed]
        successes = sum(row['result'] for row in current)
        self.totalGraspsLabel.setText(f'Total Grasps: {self.completed}')
        self.successfulGraspsLabel.setText(f'Successful: {successes}')
        self.failedGraspsLabel.setText(f'Failed: {self.completed - successes}')
        if self.completed == total:
            self.timer.stop()
            self.records.extend(self.pending_records)
            self.pending_records = []
            self.startSimulationButton.setEnabled(True)
            self.statusBar().showMessage(f'MOCK simulation completed: {total} attempts; records held in memory')
            self.refresh_dataset()
            self.update_overview()
            QMessageBox.information(self, 'Mock simulation completed',
                f'Generated {total} DEMO grasp records.\nOpen Dataset to inspect them.\n\n'
                'These are random examples, not real PyBullet results.')

    def refresh_dataset(self, *_args):
        filt = self.resultFilterComboBox.currentText().lower()
        rows = [
            (idx, record) for idx, record in enumerate(self.records, start=1)
            if filt.startswith('all') or (
                record['result'] == 1 if filt.startswith('success') else record['result'] == 0
            )
        ]
        table = self.graspTableWidget
        table.setRowCount(len(rows))
        for row_number, (record_number, record) in enumerate(rows):
            values = [record_number, record['x'], record['y'], record['z'],
                      record['roll'], record['pitch'], record['yaw'], record['result']]
            for col, value in enumerate(values):
                table.setItem(row_number, col, QTableWidgetItem(str(value)))
        self.datasetCountLabel.setText(f'Total Records: {len(rows)} (all: {len(self.records)})')

    def update_overview(self):
        total = len(self.records)
        successes = sum(r['result'] for r in self.records)
        rate = f'{successes / total * 100:.1f}%' if total else '--'
        self.dashboardTotalGraspsLabel.setText(f'Total Grasps: {total}')
        self.dashboardSuccessRateLabel.setText(f'Grasp Success Rate: {rate}')
        self.reportTotalGraspsLabel.setText(f'Total Grasps: {total}')
        self.reportSuccessRateLabel.setText(f'Grasp Success Rate: {rate}')
        self.recentActivityLabel.setText(
            f'Mock simulation completed: {total} total records (demo data)' if total else 'No recent activity.'
        )

    def logout(self):
        self.timer.stop()
        self.login_window.passwordInput.clear()
        self.login_window.show()
        self.close()


class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi(str(BASE_DIR / 'login.ui'), self)
        self.main_window = None
        self.loginButton.clicked.connect(self.login)
        self.passwordInput.returnPressed.connect(self.login)

    
    def login(self):
        username = self.usernameInput.text().strip()
        password = self.passwordInput.text()

        # Demo accounts
        if username in ["student", "jenny", "dylan", "summer", "jason"] and password == "1234":
            self.main_window = MainWindow(self, username)
            self.main_window.show()
            self.hide()
        else:
            QMessageBox.warning(
                self,
                "Login failed",
                "Invalid username or password. Please try again."
            )



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LoginWindow()
    window.show()
    sys.exit(app.exec())
