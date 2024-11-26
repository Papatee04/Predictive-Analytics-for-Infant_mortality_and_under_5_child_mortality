document.addEventListener('DOMContentLoaded', function () {
    const settingsBtn = document.getElementById('settings-dropdown-btn');
    const settingsDropdown = document.getElementById('settings-dropdown');
    const profileBtn = document.getElementById('profile-dropdown-btn');
    const profileDropdown = document.getElementById('profile-dropdown');

    // Settings Dropdown
    settingsBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        settingsDropdown.classList.toggle('hidden');
        profileDropdown.classList.add('hidden');
    });

    // Profile Dropdown
    profileBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        profileDropdown.classList.toggle('hidden');
        settingsDropdown.classList.add('hidden');
    });

    // Close dropdowns when clicking outside
    document.addEventListener('click', function () {
        settingsDropdown.classList.add('hidden');
        profileDropdown.classList.add('hidden');
    });

    // Prevent dropdown from closing when clicking inside
    settingsDropdown.addEventListener('click', function (e) {
        e.stopPropagation();
    });
    profileDropdown.addEventListener('click', function (e) {
        e.stopPropagation();
    });
});